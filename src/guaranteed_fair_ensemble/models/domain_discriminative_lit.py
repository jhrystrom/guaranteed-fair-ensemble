from collections.abc import Mapping
from typing import Any

import lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score, roc_auc_score

from guaranteed_fair_ensemble.models.domain_discriminative import (
    DomainDiscriminativeModel,
)
from guaranteed_fair_ensemble.torch_utils import reverse_one_hot

__all__ = ["DomainDiscriminativeLitModule"]


class DomainDiscriminativeLitModule(pl.LightningModule):
    """Lightning wrapper around :class:`DomainDiscriminativeModel`."""

    def __init__(
        self,
        model: DomainDiscriminativeModel,
        lr: float = 1e-4,
        scaling: float = 1.0,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

        # Hyperparameters for checkpointing / logging
        self.save_hyperparameters(ignore=["model"])

        self.num_domains = model.num_domains
        self.num_classes = model.num_classes

    # Criterion - domain-discriminative BCE
    def criterion(
        self, logits: torch.Tensor, target: torch.Tensor, domain_idx: torch.Tensor
    ) -> torch.Tensor:
        """Binary-cross-entropy where only *one* domain-head carries the label.

        * For the sample's *true* domain we supervise with the real label.
        * All other domain-heads are forced toward 0 (negative).
        """

        b, _ = logits.shape
        logits_3d = logits.view(b, self.num_domains, self.num_classes)

        # Construct per-sample targets  (B, K, C)
        tgt = torch.zeros_like(logits_3d)
        # broadcast domain index over class dimension
        tgt[torch.arange(b), domain_idx] = target.unsqueeze(-1)

        return F.binary_cross_entropy_with_logits(logits_3d, tgt)

    # Forward helpers
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.predict(x)

    # Steps
    def _shared_step(self, batch: Any, stage: str) -> Mapping[str, torch.Tensor]:
        x, y_raw, _ = batch
        target = y_raw[:, 0]  # (B,)
        domain_idx = reverse_one_hot(y_raw[:, 1:]).long()  # (B,)

        logits = self.model(x)
        loss = self.criterion(logits, target, domain_idx)

        # Inference (prob-sum) for metrics
        probs = self.model.inference_sum_prob(
            logits, self.num_domains, self.num_classes
        )

        # CPU numpy for sklearn metrics
        y_np = target.detach().cpu().numpy()
        p_np = probs.detach().cpu().numpy()

        # Handle degenerate batches (all one class)
        try:
            auc = roc_auc_score(y_np, p_np)
        except ValueError:
            auc = float("nan")

        # Recall @ threshold 0.5 - overall & per group
        preds_bin = (p_np >= 0.5).astype(int)
        try:
            recall = recall_score(y_np, preds_bin)
        except ValueError:
            recall = float("nan")

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_auc": torch.tensor(auc, device=self.device),
            f"{stage}_recall": torch.tensor(recall, device=self.device),
        }

        # Per-group recall
        for g in range(self.num_domains):
            mask = domain_idx == g
            if mask.any():
                try:
                    recall_g = recall_score(
                        y_np[mask.cpu().numpy()], preds_bin[mask.cpu().numpy()]
                    )
                except ValueError:
                    recall_g = float("nan")
                metrics[f"{stage}_recall_group_{g}"] = torch.tensor(
                    recall_g, device=self.device
                )

        return {"loss": loss, "metrics": metrics}

    # training / validation / test
    def training_step(self, batch: Any, batch_idx: int):  # noqa: ARG002
        result = self._shared_step(batch, "train")
        self.log_dict(result["metrics"], on_step=True, on_epoch=True, prog_bar=True)
        return result

    def validation_step(self, batch: Any, batch_idx: int):  # noqa: ARG002
        result = self._shared_step(batch, "val")
        self.log_dict(result["metrics"], on_step=False, on_epoch=True, prog_bar=True)
        return result

    def test_step(self, batch: Any, batch_idx: int):  # noqa: ARG002
        result = self._shared_step(batch, "test")
        self.log_dict(result["metrics"], on_step=False, on_epoch=True)
        return result

    # Optimiser
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def predict_from_features(
    model: DomainDiscriminativeLitModule, features: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        logits = model.model.classifier(features)
        return model.model.inference_sum_prob(
            logits, model.num_domains, model.num_classes
        )

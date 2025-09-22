from collections.abc import Mapping
from typing import Any

import lightning as pl
import torch
import torch.nn.functional as F
from fairret.loss import NormLoss
from fairret.statistic import TruePositiveRate
from sklearn.metrics import recall_score, roc_auc_score

from guaranteed_fair_ensemble.models.fairret_one_head import OneHeadFairretModel


class OneHeadFairretLit(pl.LightningModule):
    """Lightning wrapper around :class:`OneHeadFairretModel`."""

    def __init__(
        self,
        model: OneHeadFairretModel,
        lr: float = 1e-4,
        scaling: float = 1.0,
        use_fairret: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

        statistic = TruePositiveRate()
        self.fairret_loss_fn = NormLoss(statistic)
        self.scaling = scaling
        self.use_fairret = use_fairret

        self.save_hyperparameters(ignore=["model"])

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.predict(x)

    # -----------------------------------------------------------------
    def _shared_step(self, batch: Any, stage: str) -> Mapping[str, torch.Tensor]:
        x, y_raw, _ = batch  # sens may be one-hot or numeric

        target = y_raw[:, 0]
        sens = y_raw[:, 1:]

        logits = self.model(x)
        bce_loss = F.binary_cross_entropy_with_logits(logits, target.unsqueeze(1))
        try:
            fairret_loss = (
                self._fairret_safe(logits, sens, target) if self.use_fairret else 0.0
            )
        except NotImplementedError:
            print(f"{target=} causes issues")
            raise
        loss = bce_loss + self.scaling * fairret_loss

        probs = torch.sigmoid(logits).squeeze(1)

        y_np = target.detach().cpu().numpy()
        p_np = probs.detach().cpu().numpy()

        try:
            auc = roc_auc_score(y_np, p_np)
        except ValueError:
            auc = float("nan")

        preds_bin = (p_np >= 0.5).astype(int)
        try:
            recall = recall_score(y_np, preds_bin)
        except ValueError:
            recall = float("nan")

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_bce": bce_loss,
            f"{stage}_fairret": fairret_loss,
            f"{stage}_auc": torch.tensor(auc, device=self.device),
            f"{stage}_recall": torch.tensor(recall, device=self.device),
        }

        return {"loss": loss, "metrics": metrics}

    def training_step(self, batch: Any, batch_idx: int):  # noqa: ARG002
        out = self._shared_step(batch, "train")
        self.log_dict(out["metrics"], on_step=True, on_epoch=True, prog_bar=True)
        return out

    def validation_step(self, batch: Any, batch_idx: int):  # noqa: ARG002
        out = self._shared_step(batch, "val")
        self.log_dict(out["metrics"], on_step=False, on_epoch=True, prog_bar=True)
        return out

    def test_step(self, batch: Any, batch_idx: int):  # noqa: ARG002
        out = self._shared_step(batch, "test")
        self.log_dict(out["metrics"], on_step=False, on_epoch=True)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # --- inside OneHeadFairretLit --------------------------------------------

    def _fairret_safe(
        self,
        logits: torch.Tensor,
        sens: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Wrapper around ``self.fairret_loss_fn`` that returns *0*
        (no penalty, no gradient) if FairRet cannot be computed for
        the current batch.

        Why 0?  • keeps shapes consistent
                • produces zero-gradients -> harmless for optimisation
        """
        if not self.use_fairret:
            return torch.zeros(1, device=logits.device, dtype=logits.dtype)

        try:
            return self.fairret_loss_fn(logits, sens, target.unsqueeze(1))
        except NotImplementedError as err:
            if "Target statistic is zero" not in str(err):
                # unknown failure → surface it
                raise
            # known edge case → neutral penalty
            self.log(  # <-- optional: monitor how often this happens
                "fairret_bypass",
                torch.tensor(1.0, device=self.device),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            return torch.zeros(1, device=logits.device, dtype=logits.dtype)


def predict_from_features(
    model: OneHeadFairretLit, features: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        logits = model.model.classifier(features)
        return torch.sigmoid(logits)

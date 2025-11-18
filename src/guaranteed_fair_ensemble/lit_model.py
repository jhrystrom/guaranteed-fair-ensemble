import lightning as L
import torch
from torch import nn, optim
from torchvision.models import EfficientNet, MobileNetV3


class LitMultiHead(L.LightningModule):
    """
    Lightning module for training a multi-head classifier built on top of
    torchvision EfficientNet / MobileNetV3 backbones.

    Features:
      - Vectorized BCEWithLogits loss across all heads
      - Majority-vote metrics with configurable tie-breaking strategy
      - Optional freezing of the backbone via requires_grad flags
      - Compatible with batches shaped (x, y) or (x, y, extra)
    """

    def __init__(
        self,
        model: EfficientNet | MobileNetV3,
        lr: float = 1e-4,
        freeze_backbone: bool = True,
        log_per_head_val_loss: bool = True,
        tie_strategy: str = "mean_prob",  # 'mean_prob' | 'positive' | 'negative'
    ):
        """
        Args:
            model: A torchvision EfficientNet or MobileNetV3 with `.features`, `.avgpool`, `.classifier`.
                   The classifier's last Linear must have out_features == number of heads.
            lr: Learning rate for Adam.
            freeze_backbone: If True, freezes the feature extractor (no gradients).
            log_per_head_val_loss: If True, logs `val_loss_head{i}` for each head.
            tie_strategy: How to break ties in majority vote when number of heads is even:
                - 'mean_prob' (recommended): use mean probability > 0.5
                - 'positive': default to class 1
                - 'negative': default to class 0
        """
        super().__init__()

        # Backbone & head
        self.features = model.features
        self.avg_pool = model.avgpool
        self.classifier = model.classifier

        # Infer number of heads from the last Linear
        if not isinstance(self.classifier, nn.Sequential) or len(self.classifier) == 0:
            raise TypeError(
                "Expected `model.classifier` to be an nn.Sequential with a final nn.Linear."
            )
        last = self.classifier[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError(
                f"Expected the last layer of `model.classifier` to be nn.Linear, got {type(last)}"
            )
        self.num_members: int = last.out_features

        # Hyperparameters / options
        self.lr = lr
        self.freeze_backbone = freeze_backbone
        self.log_per_head_val_loss = log_per_head_val_loss
        self.tie_strategy = tie_strategy

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Let Lightning track hyperparameters (but not the large `model` object itself)
        self.save_hyperparameters(ignore=["model"])

    # ---- Lightning lifecycle -------------------------------------------------

    def setup(self, stage: str | None = None):  # noqa: ARG002
        """Freeze/unfreeze backbone cleanly by toggling requires_grad."""
        if self.freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
            # Some models expose parameters in avg_pool (often not trainable, but set to be safe)
            for p in self.avg_pool.parameters():
                p.requires_grad = False

    # ---- Model forward -------------------------------------------------------

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns flattened feature vectors: shape (B, F).
        Freezing is controlled via `requires_grad`; no torch.no_grad() here.
        """
        x = self.features(images)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: images tensor of shape (B, C, H, W)
        Returns:
            logits: tensor of shape (B, H) where H = number of heads.
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits

    # ---- Majority voting for metrics ----------------------------------------

    def _majority_vote(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Majority voting on per-head logits.

        Args:
            logits: (B, H)

        Returns:
            preds: (B,) float tensor in {0.0, 1.0}
        """
        H = logits.size(1)
        probs = torch.sigmoid(logits)  # (B, H)
        votes = (probs > 0.5).float()  # (B, H)
        vote_counts = votes.sum(dim=1)  # (B,)

        half = H / 2.0
        gt = vote_counts > half

        preds = torch.zeros_like(vote_counts)
        preds[gt] = 1.0
        return preds

    # ---- Shared step logic ---------------------------------------------------

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """
        Shared logic for train/val/test steps.
        Computes vectorized BCE loss across heads and majority-vote accuracy.
        """
        x, y, _ = batch

        logits = self(x)  # (B, H)

        # Targets: (B, 1) -> broadcast to (B, H)
        target = y[:, 0].float().unsqueeze(1).expand_as(logits)
        loss = self.criterion(logits, target)

        # Majority-vote metric
        preds = self._majority_vote(logits)  # (B,)
        acc = (preds == y[:, 0].float()).float().mean()

        # Logging
        on_step = stage == "train"
        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=on_step, on_epoch=True, prog_bar=True)

        # Optional per-head validation losses
        if stage == "val" and self.log_per_head_val_loss:
            for i in range(self.num_members):
                per_head_loss = self.criterion(logits[:, i], y[:, 0].float())
                # Avoid flooding the progress bar with many metrics
                self.log(
                    f"val_loss_head{i}",
                    per_head_loss,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )

        return loss

    # ---- Lightning required steps -------------------------------------------

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )


# ---------------------------------------------------------------------------
# Factory function for creating Lightning modules for different training methods
# ---------------------------------------------------------------------------


def get_lit_model_for_method(method: str, model: nn.Module, **kwargs):
    """
    Factory function to create a Lightning module appropriate for the training method.

    Args:
        method: Training method name
        model: The model to wrap (should expose .features, .avgpool, .classifier)
        **kwargs: Additional arguments for specific Lightning modules

    Returns:
        Lightning module appropriate for the specified method
    """
    if method == "erm_ensemble":
        # Backwards-compatible alias to the multi-head ERM baseline
        return LitMultiHead(model, **kwargs)

    if method == "domain_independent":
        from guaranteed_fair_ensemble.models.domain_independent_lit import (
            DomainIndependentLitModule,
        )

        return DomainIndependentLitModule(model, **kwargs)

    if method == "domain_discriminative":
        from guaranteed_fair_ensemble.models.domain_discriminative_lit import (
            DomainDiscriminativeLitModule,
        )

        return DomainDiscriminativeLitModule(model, **kwargs)

    if method == "fairret":
        from guaranteed_fair_ensemble.models.fairret_lit import OneHeadFairretLit

        return OneHeadFairretLit(model, **kwargs)

    if method in {"erm", "rebalance"}:
        from guaranteed_fair_ensemble.models.fairret_lit import OneHeadFairretLit

        return OneHeadFairretLit(model, use_fairret=False, **kwargs)

    if method == "ensemble_multi_head":
        from guaranteed_fair_ensemble.models.fairensemble_lit import FairEnsemble

        training_mask = kwargs.get("training_mask")
        if training_mask is None:
            raise ValueError(
                "Missing `training_mask` argument for ensemble_multi_head method"
            )

        # Pass through any required args; consistency handled in FairEnsemble
        return FairEnsemble(model=model, **kwargs)

    raise ValueError(
        f"Unknown training method '{method}' for Lightning module factory."
    )

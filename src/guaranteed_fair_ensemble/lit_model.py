from typing import NamedTuple

import lightning as L
import torch
from torch import nn, optim


class LossOutput(NamedTuple):
    full_loss: torch.Tensor
    classification_loss: torch.Tensor
    protected_loss: torch.Tensor


def total_loss(pred: torch.Tensor, true: torch.Tensor, scaling: float) -> LossOutput:
    """
    Combined loss function for multi-head model
    First head uses BCE loss for binary classification
    Remaining heads use MSE loss for protected attribute prediction

    Args:
        pred: Model predictions
        true: Ground truth labels
        scaling: Weight for the protected attribute loss

    Returns:
        Combined loss value
    """
    num_heads = pred.shape[1]
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    # Task loss (binary classification)
    classification_loss = bce(pred[:, 0], true[:, 0])

    # Protected attribute loss
    protected_loss = 0
    for i in range(1, num_heads):
        protected_loss += mse(pred[:, i], true[:, i])

    full_loss = classification_loss + scaling * protected_loss
    return LossOutput(
        full_loss=full_loss,
        classification_loss=classification_loss,
        protected_loss=protected_loss,
    )


class LitMultiHead(L.LightningModule):
    """
    Lightning module for standard training
    """

    def __init__(self, model: nn.Module, scaling: float = 1.0, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.scaling = scaling
        self.lr = lr
        self.train_acc = 0
        self.val_acc = 0
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        pred = self(x)
        loss, _, _ = total_loss(pred, y, self.scaling)

        # Calculate accuracy for logging
        y_pred = torch.sigmoid(pred[:, 0]) > 0.5
        y_true = y[:, 0].bool()
        acc = (y_pred == y_true).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc = acc

        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        pred = self(x)
        loss, _, _ = total_loss(pred, y, self.scaling)

        # Calculate accuracy for logging
        y_pred = torch.sigmoid(pred[:, 0]) > 0.5
        y_true = y[:, 0].bool()
        acc = (y_pred == y_true).float().mean()

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.val_acc = acc

        return loss

    def configure_optimizers(self):
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )


def get_lit_model_for_method(method: str, model: nn.Module, **kwargs):
    """
    Factory function to create a Lightning module appropriate for the training method

    Args:
        method: Training method name
        model: The model to wrap
        **kwargs: Additional arguments for specific Lightning modules

    Returns:
        Lightning module appropriate for the specified method
    """
    if method == "standard":
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
        from guaranteed_fair_ensemble.models.guaranteed_fair_ensemble_lit import (
            guaranteed_fair_ensemble,
        )

        training_mask = kwargs.get("training_mask")
        if training_mask is None:
            raise ValueError("Missing training_mask argument")
        print(f"{kwargs=}")

        # Pass the total_loss function to ensure consistency
        return guaranteed_fair_ensemble(model=model, **kwargs)

    # Placeholder for other methods - to be implemented
    print(f"Note: Specialized Lightning module for '{method}' not yet implemented.")
    print("Using standard LitMultiHead as a fallback.")
    return LitMultiHead(model, **kwargs)

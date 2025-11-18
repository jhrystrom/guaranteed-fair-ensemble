from pathlib import Path
from typing import Any, NamedTuple

import lightning as L
import torch
import torch.optim as optim
from loguru import logger
from torch import nn
from torchvision.models import EfficientNet, MobileNetV3


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


class FairEnsemble(L.LightningModule):
    def __init__(
        self,
        model: EfficientNet | MobileNetV3,
        training_mask: torch.Tensor | None = None,
        lr: float = 1e-3,
        num_heads_per_member: int = 4,
        scaling: float = 0.5,
    ):
        super().__init__()
        self.features = model.features
        self.avg_pool = model.avgpool
        self.classifiers = model.classifier
        self.lr = lr
        self.classifier_size = num_heads_per_member
        self.num_classifiers = self.classifiers[-1].out_features // num_heads_per_member
        self.scale = scaling

        # Ensure boolean mask and register as buffer so it follows device moves & checkpoints
        tm = training_mask.bool() if training_mask is not None else None
        self.register_buffer("training_mask", tm, persistent=True)

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        images, labels, indices = batch
        indices = self._indices_to_device(indices)

        features = self.extract_features(images)
        batch_loss = 0.0

        # mask: [batch, num_classifiers]
        batch_train_mask = self.training_mask[indices, :]

        for classifier_idx in range(self.num_classifiers):
            cls_mask = batch_train_mask[:, classifier_idx]

            training_examples = features[cls_mask]
            training_labels = labels[cls_mask]

            classifier_start, classifier_stop = self._get_classifier_group(
                classifier_idx
            )
            forward = self.classifiers(training_examples)[
                :, classifier_start:classifier_stop
            ]

            loss, class_loss, protect_loss = total_loss(
                pred=forward, true=training_labels, scaling=self.scale
            )
            batch_loss += loss

            # Log metrics for this fold
            self.log(
                f"train_loss_fold{classifier_idx}", loss, on_step=True, on_epoch=True
            )
            self.log(
                f"train_bce_loss_fold{classifier_idx}",
                class_loss,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                f"train_protected_loss_fold{classifier_idx}",
                protect_loss,
                on_step=True,
                on_epoch=True,
            )

        avg_batch_loss = batch_loss / self.num_classifiers
        self.log(
            "train_loss", avg_batch_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return avg_batch_loss

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        # If you want to fine-tune the backbone, remove no_grad(). If you want it frozen, keep it.
        with torch.no_grad():
            features = self.features(images)
            features = self.avg_pool(features)
            features = torch.flatten(features, 1)
        return features

    def _get_classifier_group(self, classifier_idx: int) -> tuple[int, int]:
        classifier_start = classifier_idx * self.classifier_size
        classifier_stop = classifier_start + self.classifier_size
        return classifier_start, classifier_stop

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        images, labels, indices = batch
        indices = self._indices_to_device(indices)

        features = self.extract_features(images)
        all_predictions = self.classifiers(features)

        val_loss = 0.0

        # Use logical_not on boolean mask
        batch_val_mask = ~self.training_mask[indices, :]

        for classifier_idx in range(self.num_classifiers):
            cls_mask = batch_val_mask[:, classifier_idx]

            val_labels = labels[cls_mask]
            val_predictions = all_predictions[cls_mask, :]

            classifier_start, classifier_stop = self._get_classifier_group(
                classifier_idx
            )
            # Slice the correct head for this classifier
            head_predictions = val_predictions[:, classifier_start:classifier_stop]

            fold_val_loss, val_class_loss, val_protect_loss = total_loss(
                pred=head_predictions, true=val_labels, scaling=self.scale
            )
            val_loss += fold_val_loss

            # Log metrics
            self.log(f"val_loss_fold{classifier_idx}", fold_val_loss, on_epoch=True)
            self.log(
                f"val_bce_loss_fold{classifier_idx}",
                val_class_loss,
                on_epoch=True,
            )
            self.log(
                f"val_protected_loss_fold{classifier_idx}",
                val_protect_loss,
                on_epoch=True,
            )

        avg_val_loss = val_loss / self.num_classifiers
        self.log("val_loss", avg_val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return avg_val_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifiers(features)

    def _indices_to_device(self, indices) -> torch.Tensor:
        """Ensure indices are a LongTensor on the module's current device."""
        if not torch.is_tensor(indices):
            return torch.as_tensor(indices, device=self.device, dtype=torch.long)
        return indices.to(self.device, dtype=torch.long)

    def configure_optimizers(self):
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )

    def ensemble_predict(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.extract_features(images)
            classifications = self.classifiers(features)
            # Take the first classification in each group
            probabilities = torch.sigmoid(classifications[:, 0 :: self.classifier_size])
            return probabilities


def load_guaranteed_fair_ensemble_from_checkpoint(
    ckpt_path: str | Path,
    backbone: Any,  # torchvision EfficientNet | MobileNetV3
    num_heads_per_member: int,
    device: torch.device | str = "cpu",
) -> FairEnsemble:
    """Load an guaranteed_fair_ensemble from a Lightning checkpoint with remapped keys."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state: dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)

    def _remap_key(k: str) -> str:
        # Strip Lightning nesting used during training: "model."
        if k.startswith("model."):
            k = k[len("model.") :]
        # Align attribute renames in your current module
        # torchvision uses 'avgpool' while your module uses 'avg_pool'
        k = k.replace("avgpool", "avg_pool")
        # Your module exposes 'classifiers' instead of 'classifier'
        # (keep '.classifier' to avoid renaming things like 'classifier_size')
        k = k.replace("classifier.", "classifiers.")
        return k

    remapped = {
        _remap_key(k): v for k, v in state.items() if not k.startswith("automatic_")
    }  # ignore trainer internals if present

    # Instantiate your module (this registers training_mask as a buffer)
    model = FairEnsemble(
        model=backbone,
        training_mask=None,
        num_heads_per_member=num_heads_per_member,
    )

    # Load weights with strict=False to ignore the missing 'training_mask' in checkpoint
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    # Optional: sanity logs so you can see what (if anything) didn't map
    if missing:
        logger.warning(
            f"[load] missing keys after remap (ok if only buffers): {missing}"
        )
    if unexpected:
        logger.warning(f"[load] unexpected keys after remap: {unexpected}")

    model.to(device)
    model.eval()
    return model


def predict_from_features(model: FairEnsemble, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        classifications = model.classifiers(features)
        # Take the first classification in each group
        probabilities = torch.sigmoid(classifications[:, 0 :: model.classifier_size])
        return probabilities

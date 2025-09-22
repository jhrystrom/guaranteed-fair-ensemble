# src/guaranteed_fair_ensemble/models/ensemble_lit_multihead.py
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim


class EnsembleLitMultiHead(L.LightningModule):
    """
    Lightning module for ensemble training with multiple heads
    Each ensemble member has its own set of heads that are only updated
    when that member's fold is being trained
    """

    def __init__(
        self,
        model: nn.Module,
        scaling: float,
        ensemble_members: int,
        num_heads_per_member: int,
        fold_indices: list,  # List of (train_idx, val_idx) tuples for each fold
        lr: float = 1e-4,
        loss_fn=None,  # Pass the total_loss function directly
    ):
        super().__init__()
        # Separate backbone (features + avgpool) and classifier
        self.model = model
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.scaling = scaling
        self.lr = lr
        self.ensemble_members = ensemble_members
        self.num_heads_per_member = num_heads_per_member
        self.fold_indices = fold_indices
        self.loss_fn = loss_fn  # Store the loss function

        # Convert fold indices to sets for faster lookup
        self.fold_train_sets = [set(train_idx) for train_idx, _ in fold_indices]

        # Initialize metrics
        self.train_acc_per_fold = dict.fromkeys(range(ensemble_members), 0)
        self.val_acc_per_fold = dict.fromkeys(range(ensemble_members), 0)

        self.save_hyperparameters(ignore=["model", "fold_indices", "loss_fn"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that prevents gradients in the backbone"""
        # Use no_grad for features and avgpool to ensure no gradients flow through the backbone
        with torch.no_grad():
            # Pass through features
            x = self.model.features(x)
            # Pass through avgpool
            x = self.model.avgpool(x)
            # Flatten
            x = torch.flatten(x, 1)
            # Detach to ensure complete gradient separation
            x = x.detach()

        # Run classifier with gradients enabled
        return self.model.classifier(x)

    def get_fold_masks(self, batch_indices):
        fold_masks = []
        for train_set in self.fold_train_sets:
            train_set_tensor = torch.tensor(
                list(train_set), device=batch_indices.device
            )
            mask = torch.isin(batch_indices, train_set_tensor)
            fold_masks.append(mask)
        return fold_masks

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        x, y, indices = batch

        if hasattr(self.trainer.train_dataloader.dataset, "get_fold_mask"):
            # Use the dataset's method to get fold masks - vectorized version
            fold_masks = []
            for fold_idx in range(self.ensemble_members):
                # Get the fold mask for all indices at once
                mask = self.trainer.train_dataloader.dataset.get_fold_mask(
                    indices, fold_idx
                )
                fold_masks.append(mask)
        else:
            # Fall back to original implementation
            fold_masks = self._get_fold_masks(indices)

        # Get predictions from all heads for all samples
        all_preds = self(
            x
        )  # Shape: [batch_size, ensemble_members * num_heads_per_member]

        # Initialize total batch loss
        batch_loss = 0.0

        # Process each fold
        for fold_idx, fold_mask in enumerate(fold_masks):
            if not fold_mask.any():
                # Skip if no samples in this batch belong to this fold
                continue

            # Get the samples for this fold
            fold_x = x[fold_mask]
            fold_y = y[fold_mask]

            # Get predictions for this fold's heads only
            start_head_idx = fold_idx * self.num_heads_per_member
            end_head_idx = start_head_idx + self.num_heads_per_member

            # Forward pass for this fold's samples through this fold's heads
            fold_preds = all_preds[fold_mask, start_head_idx:end_head_idx]

            # Calculate loss
            fold_loss, classification_loss, protected_loss = self.loss_fn(
                fold_preds, fold_y, self.scaling
            )

            # Accumulate loss
            batch_loss += fold_loss

            # Calculate accuracy for logging (classification head only)
            y_pred = torch.sigmoid(fold_preds[:, 0]) > 0.5
            y_true = fold_y[:, 0].bool()
            acc = (y_pred == y_true).float().mean()

            # Log metrics for this fold
            self.log(
                f"train_loss_fold{fold_idx}", fold_loss, on_step=True, on_epoch=True
            )
            self.log(
                f"train_bce_loss_fold{fold_idx}",
                classification_loss,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                f"train_protected_loss_fold{fold_idx}",
                protected_loss,
                on_step=True,
                on_epoch=True,
            )

            self.log(f"train_acc_fold{fold_idx}", acc, on_step=True, on_epoch=True)
            self.train_acc_per_fold[fold_idx] = acc.item()

        # Log average metrics across all folds
        if batch_loss > 0:  # Only if at least one fold had samples
            self.log(
                "train_loss",
                batch_loss / len(fold_masks),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "train_acc",
                sum(self.train_acc_per_fold.values()) / len(self.train_acc_per_fold),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        return batch_loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        x, y, indices = batch  # Assuming your dataloader returns sample indices

        # Similar approach for validation step
        if hasattr(self.trainer.val_dataloaders.dataset, "get_fold_mask"):
            # Use the dataset's method to get fold masks
            fold_masks = [
                torch.tensor(
                    [
                        self.trainer.val_dataloaders.dataset.is_in_fold_train(
                            idx.item(), fold_idx
                        )
                        for idx in indices
                    ],
                    device=indices.device,
                )
                for fold_idx in range(self.ensemble_members)
            ]
        else:
            # Fall back to original implementation
            fold_masks = self._get_fold_masks(indices)

        # Get all predictions
        all_preds = self(x)

        val_loss = 0.0

        # Calculate validation metrics for each fold separately
        for fold_idx, fold_mask in enumerate(fold_masks):
            # For validation, we use the validation indices instead of training indices
            # This is the opposite of the mask we used for training
            val_mask = ~fold_mask

            if not val_mask.any():
                continue

            # Get validation samples for this fold
            val_x = x[val_mask]
            val_y = y[val_mask]

            # Get predictions for this fold's heads
            start_head_idx = fold_idx * self.num_heads_per_member
            end_head_idx = start_head_idx + self.num_heads_per_member

            val_preds = all_preds[val_mask, start_head_idx:end_head_idx]

            # Calculate loss
            fold_val_loss, val_classification_loss, val_protected_loss = self.loss_fn(
                val_preds, val_y, self.scaling
            )
            val_loss += fold_val_loss

            # Calculate accuracy
            y_pred = torch.sigmoid(val_preds[:, 0]) > 0.5
            y_true = val_y[:, 0].bool()
            acc = (y_pred == y_true).float().mean()

            # Log metrics
            self.log(f"val_loss_fold{fold_idx}", fold_val_loss, on_epoch=True)
            self.log(
                f"val_bce_loss_fold{fold_idx}",
                val_classification_loss,
                on_epoch=True,
            )
            self.log(
                f"val_protected_loss_fold{fold_idx}",
                val_protected_loss,
                on_epoch=True,
            )

            self.log(f"val_acc_fold{fold_idx}", acc, on_epoch=True)
            self.val_acc_per_fold[fold_idx] = acc.item()

        # Log average metrics
        if val_loss > 0:
            self.log(
                "val_loss", val_loss / len(fold_masks), on_epoch=True, prog_bar=True
            )
            self.log(
                "val_acc",
                sum(self.val_acc_per_fold.values()) / len(self.val_acc_per_fold),
                on_epoch=True,
                prog_bar=True,
            )

        return val_loss

    def predict_step(self, batch, batch_idx):  # noqa: ARG002
        x = batch[0]
        all_preds = self(x)

        # Organize predictions by fold
        fold_predictions = []
        for fold_idx in range(self.ensemble_members):
            start_head_idx = fold_idx * self.num_heads_per_member
            end_head_idx = start_head_idx + self.num_heads_per_member
            fold_preds = all_preds[:, start_head_idx:end_head_idx]
            fold_predictions.append(fold_preds)

        return fold_predictions

    def configure_optimizers(self):
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )

import lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC

from guaranteed_fair_ensemble.models.domain_independent import DomainIndependentModel
from guaranteed_fair_ensemble.torch_utils import reverse_one_hot

Tensor = torch.Tensor


class DomainIndependentLitModule(pl.LightningModule):
    """
    PyTorch-Lightning module for Domain-Independent learning.
    """

    def __init__(
        self,
        model: DomainIndependentModel,
        scaling: float = 1.0,  # noqa: ARG002
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

        self.num_domains: int = model.num_domains
        self.num_classes: int = model.num_classes  # 1 for binary

        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()

        self.save_hyperparameters(ignore=["model"])

    # --------------------------------------------------------------------- #
    # Loss
    # --------------------------------------------------------------------- #
    def criterion_domain(
        self, output: Tensor, target: Tensor, sensitive_attr: Tensor
    ) -> Tensor:
        """
        BCE-with-logits loss computed only on the logits that correspond to
        the domain of each sample.

        Args:
            output: shape (B, D * C)
            target: shape (B,) or (B, 1) - binary labels (0/1)
            sensitive_attr: shape (B,) - domain indices in [0, D-1]
        """
        bsz, total_channels = output.shape

        # --- 1. sanity check ------------------------------------------------
        if total_channels % self.num_domains:
            raise ValueError(
                f"Model produced {total_channels} channels, which is not "
                f"a multiple of num_domains={self.num_domains}. "
                "Make sure `model` outputs D*C channels."
            )

        class_num = total_channels // self.num_domains  # C

        # --- 2. reshape & index --------------------------------------------
        # shape here is (B, D, C)
        output_3d = output.view(bsz, self.num_domains, class_num)

        # gather logits for the correct domain: (B, C)
        domain_logits = output_3d[
            torch.arange(bsz, device=output.device), sensitive_attr
        ]

        # --- 3. BCE loss ----------------------------------------------------
        target = target.float().view_as(domain_logits)
        loss = F.binary_cross_entropy_with_logits(domain_logits, target)
        return loss

    # --------------------------------------------------------------------- #
    # Forward / predict
    # --------------------------------------------------------------------- #
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def predict(self, x: Tensor) -> Tensor:
        return self.model.predict(x)

    # --------------------------------------------------------------------- #
    # Steps
    # --------------------------------------------------------------------- #
    def _shared_step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
        x, y_raw, _ = batch
        target: Tensor = y_raw[:, 0]
        sensitive_attr: Tensor = reverse_one_hot(y_raw[:, 1:])

        logits = self(x)
        loss = self.criterion_domain(logits, target, sensitive_attr)

        probs = self.predict(x)
        auc_metric = self.train_auc if stage == "train" else self.val_auc
        auc = auc_metric(probs, target.int())

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_auc", auc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        return self._shared_step(batch, "val")

    # --------------------------------------------------------------------- #
    # Optimizer
    # --------------------------------------------------------------------- #
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def predict_from_features(
    model: DomainIndependentLitModule, features: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        logits = model.model.classifier(features)
        return model.model.inference_sum_prob(
            logits, model.num_domains, model.num_classes
        )

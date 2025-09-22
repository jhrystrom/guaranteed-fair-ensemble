import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainIndependentModel(nn.Module):
    """
    Domain Independent model that learns multiple classifiers - one for each domain/sensitive attribute.
    This implements the approach from the DomainInd class in the MedFair code.
    """

    def __init__(self, backbone, num_domains=3, num_classes=1):
        """
        Initialize the Domain Independent model.

        Args:
            backbone: Backbone network (feature extractor)
            num_domains: Number of sensitive attribute domains (default: 3 for Fitzpatrick groups 1-4, 5, and 6)
            num_classes: Number of output classes per domain (default: 1 for binary classification)
        """
        super().__init__()
        self.backbone = backbone
        self.num_domains = num_domains
        self.num_classes = num_classes

        # Determine the output size of the backbone
        if hasattr(self.backbone, "fc"):
            # ResNet style model
            feature_dim = self.backbone.fc.in_features
            self._is_resnet = True
            # Replace the fully connected layer with identity
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            # MobileNet style model
            if isinstance(self.backbone.classifier, nn.Sequential):
                # Get input features from the last layer of the classifier
                feature_dim = self.backbone.classifier[-1].in_features
                self._is_resnet = False
            else:
                feature_dim = self.backbone.classifier.in_features
                self._is_resnet = False
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError("Unsupported backbone architecture")

        # Create separate classifiers for each domain
        total_classes = num_domains * num_classes
        self.classifier = nn.Linear(feature_dim, total_classes)

    def extract_features(self, x):
        """
        Extract features from the backbone

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        return self.backbone(x)

    def forward(self, x):
        """
        Forward pass through the network - returns ONLY the outputs

        Args:
            x: Input tensor

        Returns:
            outputs tensor (not a tuple)
        """
        # Extract features from the backbone
        features = self.extract_features(x)

        # Pass features through the classifier
        outputs = self.classifier(features)

        # Return just the outputs for compatibility
        return outputs

    def predict(self, x):
        """
        Predict method that handles the domain-specific logic and returns final predictions

        Args:
            x: Input tensor

        Returns:
            Final prediction probabilities after combining domains
        """
        # Extract features
        features = self.extract_features(x)

        # Pass through classifier
        outputs = self.classifier(features)

        # Sum probabilities across domains
        return self.inference_sum_prob(outputs, self.num_domains, self.num_classes)

    @staticmethod
    def inference_sum_prob(outputs, num_domains, num_classes=1):
        """
        Sum the probability from multiple domains for inference

        Args:
            outputs: Model outputs
            num_domains: Number of domains
            num_classes: Number of classes per domain

        Returns:
            Summed probabilities after sigmoid
        """
        predict_prob = outputs
        predict_prob_sum = []

        for i in range(num_domains):
            predict_prob_sum.append(
                predict_prob[:, i * num_classes : (i + 1) * num_classes]
            )

        predict_prob_sum = torch.stack(predict_prob_sum).sum(0)
        predict_prob_sum = torch.sigmoid(predict_prob_sum)

        return predict_prob_sum

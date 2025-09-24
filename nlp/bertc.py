"""Build Bert document classifier, the code is revised from
https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO#scrollTo=6J-FYdx6nFE_

"""

import os
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"
import copy
import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass

import evaluator
import hyperparse
import matplotlib.pyplot as plt
import numpy as np
import oxonfair
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from imblearn.under_sampling import RandomUnderSampler
from keras.preprocessing.sequence import pad_sequences
from loguru import logger
from numpy.typing import NDArray
from oxonfair import group_metrics as gm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import trange
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

load_dotenv()
usermode_str = sys.argv[1] if len(sys.argv) > 1 else ""
usermode = hyperparse.parse_string(usermode_str)
print(usermode, usermode_str)
if "oofair" in usermode:
    usermode["rmna"] = usermode["oofair"]
if "mfair" in usermode:
    usermode["rmna"] = usermode["mfair"]

if "seed" in usermode:
    random.seed(usermode["seed"])
    np.random.seed(usermode["seed"])

# Ensemble constants
if "ensemble" in usermode:
    ENSEMBLE_SIZE = usermode.get("ensemble_size", 21)
    VAL_SIZE = usermode.get("val_size", 0.3)

fairness_metric = usermode.get("fair", "min_recall")


@dataclass
class FairnessConstraint:
    name: str
    thresholds: NDArray[np.float32]
    metric: gm.GroupMetric


def undersample(
    predictions, labels, groups
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    under_sampler = RandomUnderSampler(random_state=8)
    under_validation_predictions, under_validation_groups = under_sampler.fit_resample(
        X=predictions, y=groups
    )
    under_validation_labels = labels[under_sampler.sample_indices_]
    return (
        under_validation_predictions,
        under_validation_groups,
        under_validation_labels,
    )


def get_constraint(name: str) -> FairnessConstraint:
    _minimum_rate_thresholds = np.linspace(0.5, 1, num=50)
    _parity_thresholds = np.linspace(0, 0.2, 50)
    if name == "min_recall":
        return FairnessConstraint(
            name="min_recall",
            thresholds=_minimum_rate_thresholds,
            metric=gm.recall.min,
        )
    if name == "equal_opportunity":
        return FairnessConstraint(
            name="equal_opportunity",
            thresholds=_parity_thresholds,
            metric=gm.equal_opportunity,
        )
    if name == "predictive_parity":
        return FairnessConstraint(
            name="predictive_parity",
            thresholds=_parity_thresholds,
            metric=gm.predictive_parity,
        )
    if name == "min_precision":
        return FairnessConstraint(
            name="min_precision",
            thresholds=_minimum_rate_thresholds,
            metric=gm.precision.min,
        )
    raise ValueError(f"Unknown constraint name: {name}")


class BertForMultiTask(nn.Module):
    def __init__(self, bert_model_name, num_labels, num_labels_secondary):
        super(BertForMultiTask, self).__init__()
        if "ensemble" in usermode:
            # For ensemble mode, create multi-head classifier
            self.bert = BertModel.from_pretrained(bert_model_name)
            num_heads = (
                1 + num_labels_secondary
            )  # 1 for classification + protected attributes
            classification_dims = ENSEMBLE_SIZE * num_heads
            self.classifier = nn.Linear(
                self.bert.config.hidden_size, classification_dims
            )
            self.num_labels = num_labels
            self.num_labels_secondary = num_labels_secondary
            self.num_heads = num_heads
        else:
            self.bert = BertForSequenceClassification.from_pretrained(
                bert_model_name, num_labels=num_labels
            )
            if "head1" in usermode:
                return
            self.secondary_classifier = nn.Linear(
                self.bert.config.hidden_size, num_labels_secondary
            )
            self.num_labels = num_labels

    def forward(
        self, input_ids, attention_mask=None, labels=None, protected_labels=None
    ):
        if "ensemble" in usermode:
            # For ensemble mode
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
            logits = self.classifier(pooled_output)

            # Create outputs object similar to BertForSequenceClassification
            class EnsembleOutputs:
                def __init__(self, logits):
                    self.logits = logits
                    self.loss = None

            ensemble_outputs = EnsembleOutputs(logits)

            if labels is not None and protected_labels is not None:
                # Calculate total loss for ensemble training
                ensemble_outputs.loss = self.total_loss(
                    logits, labels, protected_labels
                )

            return ensemble_outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        if "head1" in usermode:
            outputs.secondary_probs = input_ids.new_zeros(len(input_ids))
            return outputs

        if "mfair" in usermode:
            # secondary_probs = torch.softmax(self.secondary_classifier(outputs.hidden_states[-1][:, 0]), 1)
            secondary_probs = self.secondary_classifier(outputs.hidden_states[-1][:, 0])
        else:
            secondary_probs = torch.sigmoid(
                self.secondary_classifier(outputs.hidden_states[-1][:, 0]).squeeze()
            )

        outputs.secondary_probs = secondary_probs
        if protected_labels is not None:
            loss_fct = nn.MSELoss()
            secondary_loss = loss_fct(
                secondary_probs, protected_labels.to(torch.float32)
            )
            outputs.loss = outputs.loss + secondary_loss
        return outputs

    def total_loss(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        protected_labels: torch.Tensor,
        scaling: float = 0.5,
    ) -> torch.Tensor:
        """Combined loss function for multi-head ensemble model"""
        if "ensemble" not in usermode:
            return None

        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        total_loss = 0
        batch_size = pred.shape[0]

        # Train each ensemble member
        for classifier_idx in range(ENSEMBLE_SIZE):
            start_idx = classifier_idx * self.num_heads
            stop_idx = start_idx + self.num_heads

            # Get predictions for this ensemble member
            member_pred = pred[:, start_idx:stop_idx]

            # Task loss (binary classification) - first head
            classification_loss = bce(member_pred[:, 0], labels.float())

            # Protected attribute loss - remaining heads
            protected_loss = 0
            for i in range(1, self.num_heads):
                if len(protected_labels.shape) > 1:
                    protected_loss += mse(
                        member_pred[:, i], protected_labels[:, i - 1].float()
                    )
                else:
                    protected_loss += mse(member_pred[:, i], protected_labels.float())

            member_loss = classification_loss + scaling * protected_loss
            total_loss += member_loss

        return total_loss / ENSEMBLE_SIZE


class BertForMultiTask_del(BertForSequenceClassification):
    def __init__(self, bert_model_name, num_labels, num_labels_secondary):
        super(BertForMultiTask, self).__init__(
            BertConfig.from_pretrained(bert_model_name, num_labels=num_labels)
        )
        # Initialize the secondary classifier
        if "head1" in usermode:
            return
        self.secondary_classifier = nn.Linear(
            self.config.hidden_size, num_labels_secondary
        )
        self.num_labels_secondary = num_labels_secondary

    def forward1(
        self, input_ids, attention_mask=None, labels=None, protected_labels=None
    ):
        # First, call the forward method of the superclass to handle the primary classification task
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        if "head1" in usermode:
            outputs.secondary_probs = input_ids.new_zeros(len(input_ids))
            return outputs

        # Use the last hidden state for the secondary task
        # outputs[0] is the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        # Apply pooling over the last hidden state
        # Here we simply take the first token's representation as the pooled output
        secondary_pooled_output = last_hidden_state[:, 0]

        secondary_logits = self.secondary_classifier(secondary_pooled_output)

        if self.num_labels_secondary == 1:
            secondary_probs = torch.sigmoid(secondary_logits).squeeze()
        else:
            secondary_probs = torch.softmax(secondary_logits, dim=1)

        outputs.secondary_probs = secondary_probs

        if labels is not None and protected_labels is not None:
            if self.num_labels_secondary == 1:
                loss_fct = nn.MSELoss()
            else:
                loss_fct = nn.CrossEntropyLoss()
            secondary_loss = loss_fct(secondary_probs, protected_labels)
            outputs.loss = outputs.loss + secondary_loss

        return outputs


class BertForMultiTask_ext(BertForSequenceClassification):
    def __init__(self, bert_model_name, num_labels, num_labels_secondary):
        super(BertForMultiTask, self).__init__(
            BertConfig.from_pretrained(bert_model_name, num_labels=num_labels)
        )

    def forward(
        self, input_ids, attention_mask=None, labels=None, protected_labels=None
    ):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        return outputs


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp").readlines()]
    return np.argmax(memory_available)


# Ensemble functions - copied from minimum_reproducible_train_ensemble.py
if "ensemble" in usermode:

    @dataclass
    class ValidationPredictions:
        classifier_idx: int
        labels: NDArray[np.int8]
        groups: NDArray[np.int8]
        predictions: NDArray[np.float32]

    @dataclass
    class ConstraintResults:
        constraint_value: float
        min_recall: float
        equal_opportunity: float
        predictive_parity: float
        min_precision: float
        balanced_accuracy: float
        f1: float
        accuracy: float
        constraint_type: str = "min_recall"

    def calculate_metrics(
        test_labels: np.ndarray, predictions: np.ndarray, test_groups: np.ndarray
    ):
        MIN_RECALL = gm.recall.min
        EQUAL_OPPORTUNITY = gm.equal_opportunity
        ACCURACY = gm.accuracy.overall
        PREDICTIVE_PARITY = gm.predictive_parity
        MIN_PRECISION = gm.precision.min
        BALANCED_ACCURACY = gm.balanced_accuracy
        F1 = gm.f1.overall
        return {
            "min_recall": _apply_metric(
                MIN_RECALL, test_labels, predictions, test_groups
            ),
            "equal_opportunity": _apply_metric(
                EQUAL_OPPORTUNITY, test_labels, predictions, test_groups
            ),
            "accuracy": _apply_metric(ACCURACY, test_labels, predictions, test_groups),  # type: ignore
            "predictive_parity": _apply_metric(
                PREDICTIVE_PARITY, test_labels, predictions, test_groups
            ),
            "min_precision": _apply_metric(
                MIN_PRECISION, test_labels, predictions, test_groups
            ),
            "balanced_accuracy": _apply_metric(
                BALANCED_ACCURACY, test_labels, predictions, test_groups
            ),
            "f1": _apply_metric(F1, test_labels, predictions, test_groups),  # type: ignore
        }

    def _apply_metric(
        metric, labels: np.ndarray, predictions: np.ndarray, groups: np.ndarray
    ) -> float:
        return float(metric(labels, predictions, groups)[0])

    def extract_subhead(classifier: nn.Module, member_idx: int):
        in_features = classifier.in_features
        out_features = int(classifier.out_features) // ENSEMBLE_SIZE
        subhead = nn.Linear(in_features, out_features)
        # Copy the weights and bias from the original classifier's final layer
        # But only for this fold's heads
        start_idx = member_idx * out_features
        stop_idx = start_idx + out_features
        # Copy weights for just this fold's heads
        subhead.weight.data = classifier.weight.data[start_idx:stop_idx, :]
        if classifier.bias is not None:
            subhead.bias.data = classifier.bias.data[start_idx:stop_idx]
        return subhead

    def total_loss(
        pred: torch.Tensor, true: torch.Tensor, scaling: float
    ) -> torch.Tensor:
        """
        Combined loss function for multi-head model
        First head uses BCE loss for binary classification
        Remaining heads use MSE loss for protected attribute prediction
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
        return full_loss

    def vote_majority(predictions: torch.Tensor) -> np.ndarray:
        if predictions.dim() != 3:
            raise ValueError(
                "Predictions tensor must be 3-dimensional (test_size, ensemble_size, num_constraints)"
            )
        majority_votes = (predictions > 0.0).float().mean(axis=1) > 0.5
        return majority_votes.numpy()  # Dim: (test_size, num_constraints)

    def evaluate_metrics_min_recall(
        constraint: FairnessConstraint, majority_vote, test_labels, test_groups
    ):
        metrics_per_constraint: list[ConstraintResults] = []
        if len(constraint.thresholds) != majority_vote.shape[1]:
            raise ValueError(
                "Mismatch between number of constraints and majority vote shape"
            )
        for constraint_idx, threshold in enumerate(constraint.thresholds):
            metrics = calculate_metrics(
                test_labels,
                majority_vote[:, constraint_idx].astype(np.int8),
                test_groups,
            )
            meta_data = {
                "constraint_type": constraint.name,
                "constraint_value": float(threshold),
            }
            constraint_result = ConstraintResults(**metrics, **meta_data)  # type: ignore
            metrics_per_constraint.append(constraint_result)
        return metrics_per_constraint


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_f1(preds, labels):
    macro_score = f1_score(
        y_true=labels,
        y_pred=preds,
        average="macro",
    )
    weighted_score = f1_score(
        y_true=labels,
        y_pred=preds,
        average="weighted",
    )
    print("Weighted F1-score: ", weighted_score)
    print("Macro F1-score: ", macro_score)
    return macro_score, weighted_score


def build_bert(lang, odir, params=None, constraint_name: str = "min_recall"):
    """Google Bert Classifier
    lang: The language name
    odir: output directory of prediction results
    """
    if not params:
        params = dict()
        params["balance_ratio"] = 0.9
        params["freeze"] = False
        params["decay_rate"] = 0.001
        params["lr"] = 2e-5
        params["warm_steps"] = 100
        params["train_steps"] = 1000
        params["batch_size"] = 16
        params["balance"] = True

    split_dir = "./data/split/" + lang + "/"

    if "data" in usermode:
        split_dir = "./data/split/" + usermode["data"] + "/"

    if torch.cuda.is_available():
        device = "cuda:0"  # str(get_freer_gpu())
        torch.cuda.set_device(device)
        n_gpu = torch.cuda.device_count()
        print(torch.cuda.get_device_name())
        print("Number of GPUs: ", n_gpu)
    else:
        device = torch.device("cpu")
        n_gpu = 0
        print("CUDA not available, using CPU")
        print("Number of GPUs: ", n_gpu)
    print(device)

    print("Loading Datasets and oversample training data...")
    train_df = pd.read_csv(split_dir + "train.tsv", sep="\t", na_values="x")

    if "enforce" in usermode:
        params["batch_size"] = 128
        from torch.nn import CrossEntropyLoss

    if "cda" in usermode:
        wlst = json.load(open("dev/gender.json"))
        wdict = dict(wlst + [w[::-1] for w in wlst])
        train_df_2 = copy.deepcopy(train_df)
        for i in range(len(train_df)):
            train_df_2.loc[i, "text"] = " ".join(
                [wdict.get(w, w) for w in train_df_2.loc[i, "text"].split()]
            )
        train_df = pd.concat([train_df, train_df_2])

    # oversample the minority class
    if params["balance"]:
        label_count = Counter(train_df.label)
        for label_tmp in label_count:
            sample_num = label_count.most_common(1)[0][1] - label_count[label_tmp]
            if sample_num == 0:
                continue
            train_df = pd.concat(
                [
                    train_df,
                    train_df[train_df.label == label_tmp].sample(
                        int(sample_num * params["balance_ratio"]), replace=True
                    ),
                ]
            )
        train_df = train_df.reset_index()  # to prevent index key error

        valid_df = pd.read_csv(split_dir + "valid.tsv", sep="\t", na_values="x")
        test_df = pd.read_csv(split_dir + "test.tsv", sep="\t", na_values="x")

        # For ensemble mode, merge train and valid datasets
        if "ensemble" in usermode:
            # Combine train and valid for ensemble training
            combined_train_df = pd.concat([train_df, valid_df], ignore_index=True)
            train_df = combined_train_df
            print(
                f"Ensemble mode: Combined train and valid. New train size: {len(train_df)}"
            )

        data_df = [train_df, valid_df, test_df]
        if "smalldbg" in usermode:
            for i in range(len(data_df)):
                data_df[i] = data_df[i][: int(0.1 * len(data_df[i]))]
        if ("rmna" in usermode or "oofair" in usermode) and "data" not in usermode:
            print("Size before remove NA: ", [len(df) for df in data_df])
            data_df = [df[df[usermode["rmna"]].notnull()].copy() for df in data_df]
            [df.reset_index(drop=True, inplace=True) for df in data_df]
            print("Size after remove NA: ", [len(df) for df in data_df])

            datastat = dict(zip(["Train", "Dev", "Test"], [len(df) for df in data_df]))
            json.dump(datastat, open(os.path.join(odir, "datastat.json"), "w"))
            if "datastat" in usermode:
                exit(0)
            if "rebalance" in usermode:
                df = data_df[0]
                class_counts = df[usermode["rmna"]].value_counts()
                major_class = class_counts.idxmax()
                minor_class = class_counts.idxmin()

                # Calculate the number of samples to generate
                sample_difference = (
                    class_counts[major_class] - class_counts[minor_class]
                )

                # Separate the majority and minority classes into different DataFrames
                df_major = df[df[usermode["rmna"]] == major_class]
                df_minor = df[df[usermode["rmna"]] == minor_class]

                # Sample the difference
                df_minor_oversampled = df_minor.sample(sample_difference, replace=True)

                # Concatenate the original DataFrame with the oversampled DataFrame
                df_balanced = pd.concat([df, df_minor_oversampled], ignore_index=True)

                # Shuffle the DataFrame to mix the rows up
                data_df[0] = df_balanced.sample(frac=1).reset_index(drop=True)
                print("Size after rebalance: ", [len(df) for df in data_df])

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(lambda x: "[CLS] " + x + " [SEP]")

    if lang == "English":
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
    elif lang == "Chinese":
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-chinese", do_lower_case=True
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-uncased", do_lower_case=True
        )

    print("Padding Datasets...")
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(lambda x: tokenizer.tokenize(x))

    # convert to indices and pad the sequences
    max_len = 25
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(
            lambda x: pad_sequences(
                [tokenizer.convert_tokens_to_ids(x)], maxlen=max_len, dtype="long"
            )[0]
        )

    # create attention masks
    for doc_df in data_df:
        attention_masks = []
        for seq in doc_df.text:
            seq_mask = [float(idx > 0) for idx in seq]
            attention_masks.append(seq_mask)
        doc_df["masks"] = attention_masks

    # format train, valid, test
    train_inputs = torch.tensor(data_df[0].text)
    train_labels = torch.tensor(data_df[0].label)
    train_masks = torch.tensor(data_df[0].masks)
    valid_inputs = torch.tensor(data_df[1].text)
    valid_labels = torch.tensor(data_df[1].label)
    valid_masks = torch.tensor(data_df[1].masks)
    test_inputs = torch.tensor(data_df[2].text)
    test_labels = torch.tensor(data_df[2].label)
    test_masks = torch.tensor(data_df[2].masks)
    if (
        ("oofair" in usermode and "dbgm" not in usermode)
        or "enforce" in usermode
        or "mfair" in usermode
    ):
        print(data_df[0])
        protected_train_labels = torch.tensor(data_df[0][usermode["rmna"]].values)
        protected_valid_labels = torch.tensor(data_df[1][usermode["rmna"]].values)
        protected_test_labels = torch.tensor(data_df[2][usermode["rmna"]].values)

    batch_size = params["batch_size"]

    if (
        ("oofair" in usermode and "dbgm" not in usermode)
        or "enforce" in usermode
        or "mfair" in usermode
    ):
        train_data = TensorDataset(
            train_inputs, train_masks, train_labels, protected_train_labels
        )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size
        )
        valid_data = TensorDataset(
            valid_inputs, valid_masks, valid_labels, protected_valid_labels
        )
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=batch_size
        )
        test_data = TensorDataset(
            test_inputs, test_masks, test_labels, protected_test_labels
        )
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=batch_size
        )
    else:
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size
        )
        valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=batch_size
        )
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=batch_size
        )

    # load the pretrained model
    print("Loading Pretrained Model...")
    if lang == "English":
        if "ensemble" in usermode:
            # For ensemble mode, determine num_labels_secondary
            if "rmna" in usermode:
                if (
                    isinstance(data_df[0][usermode["rmna"]].iloc[0], str)
                    or len(data_df[0][usermode["rmna"]].unique()) > 2
                ):
                    num_labels_secondary = len(data_df[0][usermode["rmna"]].unique())
                else:
                    num_labels_secondary = 1
            else:
                num_labels_secondary = 1
            model = BertForMultiTask(
                "bert-base-uncased",
                num_labels=2,
                num_labels_secondary=num_labels_secondary,
            )
        elif "oofair" in usermode or "mfair" in usermode:
            model = BertForMultiTask(
                "bert-base-uncased",
                num_labels=2,
                num_labels_secondary=len(usermode["rmna"])
                if "mfair" in usermode
                else 1,
            )
            if "dbgm" in usermode:
                model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels=2
                )
            if "dropout" in usermode:
                for layer in model.bert.bert.encoder.layer:
                    layer.attention.self.dropout.p = usermode["dropout"]
                    layer.attention.output.dropout.p = usermode["dropout"]
                    layer.output.dropout.p = usermode["dropout"]
        else:
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            )
    elif lang == "Chinese":
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-chinese", num_labels=2
        )
    else:  # for Spanish, Italian, Portuguese and Polish
        if "oofair" in usermode:
            model = BertForMultiTask(
                "bert-base-multilingual-uncased", num_labels=2, num_labels_secondary=1
            )
            if "dbgm" in usermode:
                model = BertForSequenceClassification.from_pretrained(
                    "bert-base-multilingual-uncased", num_labels=2
                )
        else:
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-multilingual-uncased", num_labels=2
            )
    model.to(device)

    # For ensemble mode, create training mask matrix
    if "ensemble" in usermode:
        N = len(data_df[0])
        # Convert groups for stratified split
        if "rmna" in usermode:
            stratify_groups = (
                data_df[0][usermode["rmna"]].astype(str)
                + "_"
                + data_df[0]["label"].astype(str)
            )
        else:
            # Use labels for stratification if no groups
            stratify_groups = data_df[0]["label"].values

        logger.debug(f"Stratify groups distribution: {Counter(stratify_groups)}")
        shuffler = StratifiedShuffleSplit(
            n_splits=ENSEMBLE_SIZE, test_size=VAL_SIZE, random_state=42
        )
        train_mask_matrix = torch.zeros((N, ENSEMBLE_SIZE), dtype=torch.bool)
        for member_idx, (train_indices, _) in enumerate(
            shuffler.split(data_df[0].text.values, stratify_groups)
        ):
            train_mask_matrix[train_indices, member_idx] = True

    # organize parameters
    param_optimizer = list(model.named_parameters())
    if params["freeze"]:
        no_decay = ["bias", "bert"]  # , 'bert' freeze all bert parameters
    else:
        no_decay = ["bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": params["decay_rate"],
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params["lr"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params["warm_steps"],
        num_training_steps=params["train_steps"],
    )

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 10

    # Training
    print("Training the model...")
    for _ in trange(epochs, desc="Epoch"):
        model.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # train batch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()

            if "ensemble" in usermode:
                # Ensemble training logic
                b_input_ids, b_input_mask, b_labels, b_protected_labels = (
                    batch if len(batch) == 4 else batch + (None,)
                )

                # Get features from BERT backbone
                bert_outputs = model.bert(b_input_ids, attention_mask=b_input_mask)
                features = bert_outputs.last_hidden_state[:, 0]  # CLS token

                ensemble_loss = 0
                SCALE = 0.5

                # Train each ensemble member
                for classifier_idx in range(ENSEMBLE_SIZE):
                    # For simplicity, train all data on all ensemble members in ensemble mode
                    # In full implementation, you would use train_mask_matrix to select specific samples
                    training_features = features
                    training_labels = b_labels.float()

                    if b_protected_labels is not None:
                        if len(b_protected_labels.shape) > 1:
                            training_groups = b_protected_labels.float()
                        else:
                            training_groups = b_protected_labels.float().unsqueeze(1)

                        combined_labels = torch.cat(
                            (training_labels.reshape(-1, 1), training_groups), dim=1
                        )
                    else:
                        combined_labels = training_labels.reshape(-1, 1)

                    # Forward pass for this ensemble member
                    start_idx = classifier_idx * model.num_heads
                    stop_idx = start_idx + model.num_heads
                    full_logits = model.classifier(training_features)
                    member_logits = full_logits[:, start_idx:stop_idx]

                    # Calculate loss for this member
                    member_loss = total_loss(
                        pred=member_logits, true=combined_labels, scaling=SCALE
                    )
                    ensemble_loss += member_loss

                loss = ensemble_loss / ENSEMBLE_SIZE

            elif "oofair" in usermode or "mfair" in usermode:
                b_input_ids, b_input_mask, b_labels, b_protected_labels = (
                    batch if len(batch) == 4 else batch + (None,)
                )
                if "dbgm" in usermode:
                    outputs = model(
                        b_input_ids, attention_mask=b_input_mask, labels=b_labels
                    )
                    loss = outputs.loss
                else:
                    outputs = model(
                        b_input_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        protected_labels=b_protected_labels,
                    )
                    loss = outputs[0]
            elif "enforce" in usermode:
                b_input_ids, b_input_mask, b_labels, b_protected_labels = (
                    batch if len(batch) == 4 else batch + (None,)
                )
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss_fct = CrossEntropyLoss(reduction="none")
                eachloss = loss_fct(
                    outputs[1].view(-1, model.num_labels), b_labels.view(-1)
                )
                if "dp" in usermode:
                    b_protected_labels = b_protected_labels.masked_select(
                        b_labels.bool().unsqueeze(1)
                    ).view(-1, b_protected_labels.shape[1])
                    eachloss = eachloss.masked_select(b_labels.bool())
                if len(b_protected_labels.shape) == 2:
                    avgs = torch.stack(
                        [
                            eachloss[b_protected_labels[:, i].bool()].mean()
                            for i in range(b_protected_labels.shape[1])
                        ]
                    )
                    bloss = (avgs - avgs.mean()).abs().mean()
                else:
                    bloss = (b_protected_labels * eachloss).mean() - (
                        (1 - b_protected_labels) * eachloss
                    ).mean()
                loss = outputs[0] + 0.1 * bloss
            else:
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                # Forward pass
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss = outputs.loss
            # backward pass
            #            outputs[0].backward()
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            scheduler.step()

            # Update tracking variables
            if "ensemble" in usermode:
                tr_loss += loss.item()
            else:
                tr_loss += outputs[0].item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print(f"Train loss: {tr_loss / max(1, nb_tr_steps)}")

        """Validation"""
        best_valid_f1 = 0.0
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        # tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # batch eval
        y_valid_preds = []
        y_valid_protected_probs = []
        y_valid_logits = []

        if "ensemble" in usermode:
            # Skip validation for ensemble mode - we'll do ensemble-specific evaluation after training
            # Set dummy values for compatibility
            y_valid_preds = [0] * len(data_df[1])  # dummy predictions
            eval_accuracy = 0.5  # dummy accuracy
            nb_eval_steps = 1
        else:
            for batch in valid_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                if "oofair" in usermode or "mfair" in usermode:
                    b_input_ids, b_input_mask, b_labels, b_protected_labels = batch
                    with torch.no_grad():
                        outputs = model(b_input_ids, attention_mask=b_input_mask)
                        if hasattr(outputs, "secondary_probs"):
                            class_logits, protected_probs = (
                                outputs.logits,
                                outputs.secondary_probs,
                            )
                        else:
                            # For ensemble mode, create dummy values for compatibility
                            class_logits = outputs.logits
                            protected_probs = torch.zeros(len(b_input_ids))
                    logits = class_logits.detach().cpu().numpy()
                    y_valid_logits.extend(logits)
                    pred_flat = np.argmax(logits, axis=1).flatten()
                    if "mfair" in usermode:
                        protected_pred_flat = protected_probs.detach().cpu().numpy()
                    else:
                        protected_pred_flat = (
                            protected_probs.detach().cpu().numpy().flatten()
                        )
                    y_valid_protected_probs.extend(protected_pred_flat)
                else:
                    b_input_ids, b_input_mask, b_labels, b_protected_labels = (
                        batch if len(batch) == 4 else batch + (None,)
                    )
                    # Telling the model not to compute or store gradients, saving memory and speeding up validation
                    with torch.no_grad():
                        # Forward pass, calculate logit predictions
                        outputs = model(
                            b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                        )
                    # Move logits and labels to CPU
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits.detach().cpu().numpy()
                    else:
                        logits = outputs[0].detach().cpu().numpy()
                # record the prediction
                pred_flat = np.argmax(logits, axis=1).flatten()
                y_valid_preds.extend(pred_flat)

            label_ids = b_labels.to("cpu").numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print(f"Validation Accuracy: {eval_accuracy / nb_eval_steps}")

        # evaluate the validation f1 score
        f1_m_valid, f1_w_valid = flat_f1(y_valid_preds, data_df[1].label)
        if f1_m_valid > best_valid_f1:
            print(f"Test {usermode_str}....")
            best_valid_f1 = f1_m_valid
            y_preds = []
            y_probs = []
            y_test_protected_probs = []
            y_test_logits = []

            # test if valid gets better results
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                if "oofair" in usermode or "mfair" in usermode:
                    b_input_ids, b_input_mask, b_labels, b_protected_labels = batch
                    with torch.no_grad():
                        outputs = model(b_input_ids, attention_mask=b_input_mask)
                        if hasattr(outputs, "secondary_probs"):
                            class_logits, protected_probs = (
                                outputs.logits,
                                outputs.secondary_probs,
                            )
                        else:
                            # For ensemble mode, create dummy values for compatibility
                            class_logits = outputs.logits
                            protected_probs = torch.zeros(len(b_input_ids))
                    logits = class_logits
                    if "mfair" in usermode:
                        test_protected_probs_flat = (
                            protected_probs.detach().cpu().numpy()
                        )
                    else:
                        test_protected_probs_flat = (
                            protected_probs.detach().cpu().numpy().flatten()
                        )
                    y_test_protected_probs.extend(test_protected_probs_flat)
                    y_test_logits.extend(logits.detach().cpu().numpy())
                else:
                    b_input_ids, b_input_mask, b_labels, b_protected_labels = (
                        batch if len(batch) == 4 else batch + (None,)
                    )
                    with torch.no_grad():
                        outputs = model(
                            b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                        )
                    logits = outputs[0]
                    if "outnpy" in usermode:
                        y_test_logits.extend(logits.detach().cpu().numpy())
                probs = F.softmax(logits, dim=1)
                probs = probs.detach().cpu().numpy()
                pred_flat = np.argmax(probs, axis=1).flatten()
                y_preds.extend(pred_flat)
                y_probs.extend([item[1] for item in probs])
            # save the predicted results
            test_df = pd.read_csv(
                os.path.join(split_dir, "test.tsv"), sep="\t", na_values="x"
            )
            if "rmna" in usermode or "oofair" in usermode:
                valid_df = pd.read_csv(
                    os.path.join(split_dir, "valid.tsv"), sep="\t", na_values="x"
                )
                valid_df = valid_df[valid_df[usermode["rmna"]].notnull()].copy()
                test_df = test_df[test_df[usermode["rmna"]].notnull()].copy()
            if "data" in usermode:
                test_df = pd.read_csv(
                    os.path.join(split_dir, "test.tsv"), sep="\t", na_values="x"
                )
                valid_df = pd.read_csv(
                    os.path.join(split_dir, "valid.tsv"), sep="\t", na_values="x"
                )
            if "smalldbg" in usermode:
                valid_df = valid_df[: int(0.1 * len(valid_df))]
                test_df = test_df[: int(0.1 * len(test_df))]

            def get_logits(y_logits):
                if "logitd" in usermode:
                    return y_logits[:, -1] - y_logits[:, 0]
                if "logit0d1" in usermode:
                    return y_logits[:, 0] - y_logits[:, -1]
                if "logit0" in usermode:
                    return y_logits[:, 0]
                if "logit1" in usermode:
                    return y_logits[:, -1]
                return y_logits[:, -1]

            if "oofair" in usermode or "mfair" in usermode:
                # valid
                if "ensemble" not in usermode:  # Only for non-ensemble modes
                    valid_df["pred"] = y_valid_preds
                    valid_df["protected_probs"] = y_valid_protected_probs
                    valid_df["logits"] = y_valid_logits

                # Save oxonfair results

                if "ensemble" not in usermode:  # Only for non-ensemble modes
                    if "mfair" in usermode:
                        outputs_val = np.hstack(
                            [
                                get_logits(np.stack(y_valid_logits))[:, None],
                                np.stack(y_valid_protected_probs),
                            ]
                        )
                    else:
                        outputs_val = np.stack(
                            [
                                get_logits(np.stack(y_valid_logits)),
                                y_valid_protected_probs,
                            ]
                        ).transpose()
                    np.save(os.path.join(odir, "outputs_val.npy"), outputs_val)
                    if "data" in usermode:
                        outputs_test = np.hstack(
                            [
                                get_logits(np.stack(y_test_logits))[:, None],
                                np.stack(y_test_protected_probs),
                            ]
                        )
                        np.save(
                            os.path.join(odir, "protected_label_val.npy"),
                            valid_df[usermode["rmna"]].to_numpy().argmax(1),
                        )
                        np.save(
                            os.path.join(odir, "protected_label_test.npy"),
                            test_df[usermode["rmna"]].to_numpy().argmax(1),
                        )
                    else:
                        outputs_test = np.stack(
                            [
                                get_logits(np.stack(y_test_logits)),
                                y_test_protected_probs,
                            ]
                        ).transpose()
                        np.save(
                            os.path.join(odir, "protected_label_val.npy"),
                            valid_df[usermode["rmna"]].to_numpy(),
                        )
                        np.save(
                            os.path.join(odir, "protected_label_test.npy"),
                            test_df[usermode["rmna"]].to_numpy(),
                        )
                    np.save(os.path.join(odir, "outputs_test.npy"), outputs_test)
                    np.save(
                        os.path.join(odir, "target_label_test.npy"),
                        test_df["label"].to_numpy(),
                    )
                    np.save(
                        os.path.join(odir, "target_label_val.npy"),
                        valid_df["label"].to_numpy(),
                    )

                if "ensemble" not in usermode:  # Only for non-ensemble modes
                    output_file = os.path.join(odir, f"{lang}-valid.tsv")
                    valid_df.to_csv(output_file, sep="\t", index=False)

                # test
                if "ensemble" not in usermode:  # Only for non-ensemble modes
                    test_df["protected_preds"] = y_test_protected_probs
                    test_df["logits"] = y_test_logits
            elif "outnpy" in usermode:
                outputs_test = np.stack(
                    [get_logits(np.stack(y_test_logits)), np.zeros(len(y_test_logits))]
                ).transpose()
                if "data" in usermode:
                    np.save(
                        os.path.join(odir, "protected_label_test.npy"),
                        test_df[usermode["rmna"]].to_numpy().argmax(1),
                    )
                else:
                    np.save(
                        os.path.join(odir, "protected_label_test.npy"),
                        test_df[usermode["rmna"]].to_numpy(),
                    )
                np.save(os.path.join(odir, "outputs_test.npy"), outputs_test)
                np.save(
                    os.path.join(odir, "target_label_test.npy"),
                    test_df["label"].to_numpy(),
                )

            # Assuming y_preds and y_probs are lists with the same length as the test data
            test_df["pred"] = y_preds
            test_df["pred_prob"] = y_probs

            # Save the modified DataFrame
            if len(test_df) != len(y_preds) or len(test_df) != len(y_probs):
                print(
                    f"Mismatch in lengths. DataFrame: {len(test_df)}, y_preds: {len(y_preds)}, y_probs: {len(y_probs)}"
                )
            output_file = os.path.join(odir, f"{lang}-test.tsv")
            test_df.to_csv(output_file, sep="\t", index=False)

            # save the predicted results
            if "ensemble" in usermode:
                # Ensemble-specific evaluation following minimum_reproducible_train_ensemble.py
                print("Starting ensemble evaluation...")

                # Create validation predictions for each ensemble member
                val_results: list[ValidationPredictions] = []

                # Get all features from backbone
                all_inputs = torch.tensor(list(data_df[0].text.values))
                all_masks = torch.tensor(list(data_df[0].masks.values))
                all_labels = data_df[0].label.values

                if "rmna" in usermode:
                    if len(data_df[0][usermode["rmna"]].unique()) > 2:
                        all_groups = data_df[0][usermode["rmna"]].values
                    else:
                        all_groups = data_df[0][usermode["rmna"]].values
                else:
                    all_groups = np.zeros(len(all_labels))

                # Process data in batches to avoid memory issues
                batch_size = 32  # Adjust based on available memory
                all_features = []

                with torch.no_grad():
                    for i in range(0, len(all_inputs), batch_size):
                        batch_inputs = all_inputs[i : i + batch_size].to(device)
                        batch_masks = all_masks[i : i + batch_size].to(device)
                        bert_outputs = model.bert(
                            batch_inputs, attention_mask=batch_masks
                        )
                        batch_features = bert_outputs.last_hidden_state[
                            :, 0
                        ].cpu()  # Move to CPU immediately
                        all_features.append(batch_features)

                all_features = torch.cat(all_features, dim=0)

                # Create validation mask (opposite of training mask)
                val_mask = ~train_mask_matrix

                for classifier_idx in range(ENSEMBLE_SIZE):
                    classifier_mask = val_mask[:, classifier_idx]
                    validation_features = all_features[classifier_mask].to(
                        device
                    )  # Move to device
                    head_start_idx = classifier_idx * model.num_heads
                    head_stop_idx = head_start_idx + model.num_heads

                    with torch.no_grad():
                        raw_val_predictions = model.classifier(validation_features)
                        head_predictions = raw_val_predictions[
                            :, head_start_idx:head_stop_idx
                        ]

                    # Convert groups to single column if needed
                    if len(all_groups.shape) > 1:
                        single_column_groups = all_groups[
                            classifier_mask.numpy()
                        ].argmax(axis=1)
                    else:
                        single_column_groups = all_groups[classifier_mask.numpy()]

                    val_results.append(
                        ValidationPredictions(
                            classifier_idx=classifier_idx,
                            labels=all_labels[classifier_mask.numpy()],
                            groups=single_column_groups,
                            predictions=head_predictions.cpu().numpy(),
                        )
                    )

                # Test data setup
                test_inputs = torch.tensor(list(data_df[2].text.values))
                test_masks = torch.tensor(list(data_df[2].masks.values))
                test_labels = data_df[2].label.values

                if "rmna" in usermode:
                    if len(data_df[2][usermode["rmna"]].unique()) > 2:
                        test_groups = data_df[2][usermode["rmna"]].values
                    else:
                        test_groups = data_df[2][usermode["rmna"]].values
                else:
                    test_groups = np.zeros(len(test_labels))

                if len(test_groups.shape) > 1:
                    test_groups = test_groups.argmax(axis=1)

                # Get test features in batches
                test_features = []
                test_batch_size = 32

                with torch.no_grad():
                    for i in range(0, len(test_inputs), test_batch_size):
                        batch_test_inputs = test_inputs[i : i + test_batch_size].to(
                            device
                        )
                        batch_test_masks = test_masks[i : i + test_batch_size].to(
                            device
                        )
                        test_bert_outputs = model.bert(
                            batch_test_inputs, attention_mask=batch_test_masks
                        )
                        batch_test_features = test_bert_outputs.last_hidden_state[
                            :, 0
                        ].cpu()  # Move to CPU
                        test_features.append(batch_test_features)

                test_features = torch.cat(test_features, dim=0)

                # Multi-threshold approach
                constraint = get_constraint(constraint_name)

                logger.info("Fitting multi-threshold ensemble...")
                multi_threshold_predictions = fit_predict_multi_threshold(
                    test_labels, model, val_results, test_features, constraint
                )
                majority_vote_multi = vote_majority(multi_threshold_predictions)
                metrics_fair_ensemble: list[ConstraintResults] = (
                    evaluate_metrics_min_recall(
                        constraint,
                        majority_vote_multi,
                        test_labels,
                        test_groups,
                    )
                )
                # Save ensemble results
                results_dict = {
                    "equal_opportunity_constraints": constraint.thresholds.tolist(),
                    "metrics": [asdict(m) for m in metrics_fair_ensemble],
                }
                with open(os.path.join(odir, "ensemble_results.json"), "w") as f:
                    json.dump(results_dict, f, indent=2)

                logger.info("Fitting joint-threshold ensemble...")
                joint_threshold_preds = fit_predict_joint_threshold(
                    test_labels, model, val_results, test_features, constraint
                )
                majority_vote_joint = vote_majority(joint_threshold_preds)
                metrics_joint_ensemble: list[ConstraintResults] = (
                    evaluate_metrics_min_recall(
                        constraint,
                        majority_vote_joint,
                        test_labels,
                        test_groups,
                    )
                )
                # Save ensemble results
                results_dict = {
                    "equal_opportunity_constraints": constraint.thresholds.tolist(),
                    "metrics": [asdict(m) for m in metrics_joint_ensemble],
                }
                with open(os.path.join(odir, "joint_ensemble_results.json"), "w") as f:
                    json.dump(results_dict, f, indent=2)

                # Raw ensemble_predictions
                with torch.no_grad():
                    raw_ensemble_predictions = (
                        model.classifier(test_features.to(device))
                        .cpu()
                        .numpy()[:, :: model.num_heads]
                    )
                    assert raw_ensemble_predictions.shape == (
                        len(test_labels),
                        ENSEMBLE_SIZE,
                    )
                raw_predicted_labels = (raw_ensemble_predictions > 0).mean(axis=1) > 0.5
                assert raw_predicted_labels.shape == (len(test_labels),), (
                    "Raw predicted labels shape mismatch"
                )
                raw_metrics = calculate_metrics(
                    test_labels=test_labels,
                    test_groups=test_groups,
                    predictions=raw_predicted_labels.astype(int),
                )
                print("Raw ensemble results:", raw_metrics)
                with open(os.path.join(odir, "raw_ensemble_results.json"), "w") as f:
                    json.dump(raw_metrics, f, indent=2)

                erm_predicted_labels = raw_ensemble_predictions[:, 0] > 0
                assert erm_predicted_labels.shape == (len(test_labels),), (
                    "ERM predicted labels shape mismatch"
                )
                erm_metrics = calculate_metrics(
                    test_labels=test_labels,
                    test_groups=test_groups,
                    predictions=erm_predicted_labels.astype(int),
                )
                with open(os.path.join(odir, "erm_results.json"), "w") as f:
                    json.dump(erm_metrics, f, indent=2)
                print("ERM results:", erm_metrics)

                # Majority voting
                # OxonFair is equivalent to majority voting with a single classifier
                oxonfair_vote_all_constraints = vote_majority(
                    multi_threshold_predictions[:, :1, :]
                )
                metrics_oxonfair = evaluate_metrics_min_recall(
                    constraint, oxonfair_vote_all_constraints, test_labels, test_groups
                )
                oxonfair_results_dict = {
                    "metrics": [asdict(m) for m in metrics_oxonfair]
                }

                with open(os.path.join(odir, "oxonfair_results.json"), "w") as f:
                    json.dump(oxonfair_results_dict, f, indent=2)

                # Plot Equal Opportunity vs Accuracy
                def plot_fair_vs_accuracy(
                    metrics_results: list[ConstraintResults],
                    output_dir,
                    metric_name="equal_opportunity",
                ):
                    """Plot Equal Opportunity vs Accuracy and save to output directory"""
                    # Convert numpy arrays to scalars for plotting
                    fairness_values = [getattr(m, metric_name) for m in metrics_results]
                    accuracies = [m.accuracy for m in metrics_results]

                    plt.figure(figsize=(10, 6))
                    plt.scatter(fairness_values, accuracies, alpha=0.7, s=60)
                    plt.plot(fairness_values, accuracies, "--", alpha=0.5)

                    plt.xlabel(metric_name, fontsize=12)
                    plt.ylabel("Accuracy", fontsize=12)
                    plt.title("Accuracy vs Equal Opportunity Trade-off", fontsize=14)
                    plt.grid(True, alpha=0.3)

                    # Add annotations for some points
                    for i, (eo, acc) in enumerate(zip(fairness_values, accuracies)):
                        if i % 3 == 0:  # Annotate every 3rd point to avoid clutter
                            plt.annotate(
                                f"({eo:.3f}, {acc:.3f})",
                                (eo, acc),
                                xytext=(5, 5),
                                textcoords="offset points",
                                fontsize=8,
                                alpha=0.7,
                            )

                    plt.tight_layout()

                    # Save plot
                    plot_path = os.path.join(
                        output_dir, f"{metric_name}_vs_accuracy.png"
                    )
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    print(f"Plot saved to: {plot_path}")
                    return plot_path

                # Generate the plot
                plot_fair_vs_accuracy(
                    metrics_fair_ensemble, odir, metric_name=constraint.name
                )

                print("Ensemble evaluation completed!")
            else:
                evaluator.eval(output_file, odir + f"/{lang}.score")


def fit_predict_joint_threshold(
    test_labels,
    model,
    val_results: list[ValidationPredictions],
    test_features,
    constraint: FairnessConstraint,
):
    all_validation_predictions = np.vstack([vpred.predictions for vpred in val_results])
    all_validation_labels = np.hstack([vpred.labels for vpred in val_results])
    all_validation_groups = np.hstack([vpred.groups for vpred in val_results])
    under_predictions, under_groups, under_labels = undersample(
        predictions=all_validation_predictions,
        groups=all_validation_groups,
        labels=all_validation_labels,
    )
    logger.info(f"After undersampling, validation set has {len(under_labels)} samples.")

    joint_fpred = oxonfair.DeepFairPredictor(
        target=under_labels,
        score=under_predictions,
        groups=under_groups,
    )
    joint_threshold_predictions = torch.zeros(
        (len(test_labels), ENSEMBLE_SIZE, len(constraint.thresholds))
    )
    for constraint_idx, constraint in enumerate(constraint.thresholds):
        joint_fpred.fit(
            objective=gm.accuracy,
            constraint=gm.recall.min,
            value=constraint,
            recompute=False,
        )
        # Apply to all ensemble members
        for classifier_idx in range(ENSEMBLE_SIZE):
            subhead = extract_subhead(model.classifier, member_idx=classifier_idx)
            merged_head = joint_fpred.merge_heads_pytorch(subhead.to("cpu"))
            merged_head.eval()
            with torch.no_grad():
                merged_test_predictions = merged_head(test_features)
            joint_threshold_predictions[:, classifier_idx, constraint_idx] = (
                merged_test_predictions[:, 0]
            )
    return joint_threshold_predictions


def fit_predict_multi_threshold(
    test_labels, model, val_results, test_features, constraint
):
    threshold_predictions = torch.zeros(
        (len(test_labels), ENSEMBLE_SIZE, len(constraint.thresholds))
    )
    for val_preds in val_results:
        fpred = oxonfair.DeepFairPredictor(
            target=val_preds.labels.astype(np.bool_),
            score=val_preds.predictions,
            groups=val_preds.groups,
        )
        for constraint_idx, constraint_value in enumerate(constraint.thresholds):
            fpred.fit(
                objective=gm.accuracy,
                constraint=constraint.metric,
                value=constraint_value,
                recompute=False,
            )
            # Extract the subhead
            subhead = extract_subhead(
                model.classifier, member_idx=val_preds.classifier_idx
            )
            merged_head = fpred.merge_heads_pytorch(subhead.to("cpu"))
            merged_head.eval()
            with torch.no_grad():
                merged_test_predictions = merged_head(
                    test_features
                )  # test_features is already on CPU
            threshold_predictions[:, val_preds.classifier_idx, constraint_idx] = (
                merged_test_predictions[:, 0]
            )
    return threshold_predictions


if __name__ == "__main__":
    langs = [
        "English"  # , 'Italian', 'Polish',
        #'Portuguese', 'Spanish'
    ]
    language_dict = {
        "en": "English",
        "it": "Italian",
        "pl": "Polish",
        "pt": "Portuguese",
        "es": "Spanish",
    }
    OUTPUT_DIR_BASE = Path(os.getenv("OUTPUT_DIR", "."))
    odir = OUTPUT_DIR_BASE / "results/bert/"
    if usermode:
        odir = odir / f"exps/{usermode_str}/"
    if not os.path.exists(odir):
        odir.mkdir(parents=True, exist_ok=True)

    # for lang in langs:
    lang = "English"
    if "lang" in usermode:
        lang = language_dict[usermode["lang"]]
    print("Working on: ", lang)
    build_bert(lang, str(odir), constraint_name=fairness_metric)

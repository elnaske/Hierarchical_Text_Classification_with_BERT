import os
import numpy as np
import torch
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics_t1(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': accuracy,
        'f1-score': f1
    }

def compute_metrics(eval_pred):
    """
    Calculates accuracy and F1-score for all three tiers.
    """
    predictions, labels = eval_pred
    tier1_logits, tier2_logits, tier3_logits = predictions
    tier1_labels, tier2_labels, tier3_labels = labels

    tier1_preds = np.argmax(tier1_logits, axis=-1)
    tier2_preds = np.argmax(tier2_logits, axis=-1)
    tier3_preds = np.argmax(tier3_logits, axis=-1)

    tier1_acc = accuracy_score(tier1_labels, tier1_preds)
    tier2_acc = accuracy_score(tier2_labels, tier2_preds)
    tier3_acc = accuracy_score(tier3_labels, tier3_preds)

    tier1_f1 = f1_score(tier1_labels, tier1_preds, average = 'macro')
    tier2_f1 = f1_score(tier2_labels, tier2_preds, average = 'macro')
    tier3_f1 = f1_score(tier3_labels, tier3_preds, average = 'macro')

    return {
        "Tier 1 Accuracy": tier1_acc,
        "Tier 2 Accuracy": tier2_acc,
        "Tier 3 Accuracy": tier3_acc,
        "Tier 1 F1": tier1_f1,
        "Tier 2 F1": tier2_f1,
        "Tier 3 F1": tier3_f1
    }

def get_mask_matrix(relations_map, parent_idxs, child_idxs):
    """
    Parses the parent-child relationships between two tiers of the hierarchy and turns them into a binary matrix.
    Rows represent the higher tier.
    Columns represents the lower tier.
    The values within the matrix are binary, where 1 represents the existance and 0 the absence of a relationship.
    """
    mask = torch.zeros((len(parent_idxs), len(child_idxs)))

    for parent, children in relations_map.items():
        for child in children:
            p_idx = parent_idxs[parent]
            c_idx = child_idxs[child]
            mask[p_idx, c_idx] = 1
    return mask

class HierarchicalDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Separate inputs and labels
        inputs = [{k: v for k, v in f.items() if k not in ["tier1", "tier2", "tier3"]} for f in features]
        tier1_labels = torch.tensor([f["t1_labels"] for f in features])
        tier2_labels = torch.tensor([f["t2_labels"] for f in features])
        tier3_labels = torch.tensor([f["t3_labels"] for f in features])

        # Use the parent class to pad input fields
        batch = super().__call__(inputs)
        batch["t1_labels"] = tier1_labels
        batch["t2_labels"] = tier2_labels
        batch["t3_labels"] = tier3_labels

        return batch

def save_model(save_name, model):
    dir = 'models'
    save_path = os.path.join(dir, save_name)
    model_save_path = os.path.join(save_path, "model")

    model.save_pretrained(model_save_path)
    print(f'Model saved to {model_save_path}')
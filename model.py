import torch
from torch import nn
from transformers import RobertaForSequenceClassification

class HierarchicalBERT(RobertaForSequenceClassification):
    def __init__(self, config, num_tier2, num_tier3, mask_tier2, mask_tier3):
        super().__init__(config)

        self.t2_clf = nn.Linear(config.hidden_size, num_tier2)
        self.t3_clf = nn.Linear(config.hidden_size, num_tier3)

        self.sm = nn.Softmax(dim = -1)

        self.mask_mat2 = mask_tier2
        self.mask_mat3 = mask_tier3

        self.init_weights()

    def forward(self, input_ids, attention_mask, t1_labels, t2_labels, t3_labels, predict = False):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=t1_labels, output_hidden_states = True)
        t1_logits = outputs.logits
        cls_output = outputs.hidden_states[-1][:, 0, :]

        t2_mask = self.get_mask(t1_logits, self.mask_mat2)

        t2_logits = self.t2_clf(cls_output)
        t2_logits_masked = t2_logits * t2_mask

        t3_mask = self.get_mask(t2_logits_masked, self.mask_mat3)

        t3_logits = self.t3_clf(cls_output)
        t3_logits_masked = t3_logits * t3_mask

        outputs = {
            "tier1_logits": t1_logits,
            "tier2_logits": t2_logits_masked,
            "tier3_logits": t3_logits_masked
        }

        if not predict:
            loss_fn = nn.CrossEntropyLoss()
            tier1_loss = loss_fn(t1_logits, t1_labels)
            tier2_loss = loss_fn(t2_logits, t2_labels)
            tier3_loss = loss_fn(t3_logits, t3_labels)
            outputs["loss"] = tier1_loss + tier2_loss + tier3_loss

        return outputs

    def get_mask(self, logits, mask_mat):
        """
        Gets the mask for the following tier based on the prediction.
        Args:
            logits: output of the prediction layer
            mask_mat: matrix containing the mask for all parent-child combinations. Rows are parents, columns are children
        Returns:
            mask: vector containing the possible classes for the next tier
        """
        sm = self.sm(logits)
        am = sm.argmax(dim = 1)

        mask = mask_mat[am.cpu()]

        return mask.to("cuda" if torch.cuda.is_available() else "cpu")
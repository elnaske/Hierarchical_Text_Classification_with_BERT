from transformers import RobertaTokenizer, Trainer
import torch
from torch import nn

from dataset import parse_dataset, get_classes, Tier3Dataset
from trainer_utils import compute_metrics, get_mask_matrix, HierarchicalDataCollator
from model import HierarchicalBERT

def main():
    test_set = parse_dataset('/data/BlurbGenreCollection_EN_test.txt')
    
    classes, class_to_idx, idx_to_class, taxonomy = get_classes(test_set)

    mask_mat_t2 = get_mask_matrix(taxonomy['tier1'], class_to_idx['tier1'], class_to_idx['tier2'])
    mask_mat_t3 = get_mask_matrix(taxonomy['tier1'], class_to_idx['tier2'], class_to_idx['tier3'])

    model = HierarchicalBERT.from_pretrained('model',
                                             num_labels = len(classes['tier1']),
                                             num_tier2 = len(classes['tier2']),
                                             num_tier3 = len(classes['tier3']),
                                             mask_tier2 = mask_mat_t2,
                                             mask_tier3 = mask_mat_t3)
    
    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

    test_dataset = Tier3Dataset(test_set['description'], test_set['tier1'], test_set['tier2'], test_set['tier3'], tokenizer)

    data_collator = HierarchicalDataCollator(tokenizer)

    tester = Trainer(
    model = model,
    data_collator = data_collator,
    compute_metrics = compute_metrics
    )

    tester.evaluate(test_dataset)

if __name__=="__main__":
    main()
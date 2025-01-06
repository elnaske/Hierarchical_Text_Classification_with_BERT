from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
from torch import nn

from dataset import download_dataset, parse_dataset, get_classes, Tier1Dataset, Tier3Dataset
from trainer_utils import compute_metrics_t1, compute_metrics, get_mask_matrix, save_model, HierarchicalDataCollator
from model import HierarchicalBERT

def train_t1(train_set, eval_set):
    classes, class_to_idx, idx_to_class, _ = get_classes(train_set)

    model_name = "FacebookAI/roberta-base"
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels = len(classes['tier1']))
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    y_train = [class_to_idx[c] for c in train_set['tier1']]
    y_eval = [class_to_idx[c] for c in eval_set['tier1']]

    print('Tokenizing data...')
    train_dataset_t1 = Tier1Dataset(train_set['description'], y_train, tokenizer)
    eval_dataset_t1 = Tier1Dataset(eval_set['description'], y_eval, tokenizer)
    print('Tokenization finished')

    training_args = TrainingArguments(output_dir = 'roberta-t1-genres',
                                      eval_strategy = 'epoch',
                                      save_strategy = 'epoch')
    
    trainer_t1 = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset_t1,
        eval_dataset = eval_dataset_t1,
        compute_metrics = compute_metrics_t1
        )
    
    trainer_t1.train()

    model.config.id2label = idx_to_class['tier1']
    model.config.label2id = class_to_idx['tier1']

    return model, tokenizer

def train(train_set, eval_set):
    classes, class_to_idx, idx_to_class, taxonomy = get_classes(train_set)

    # Convert the taxonomy to matrices designating the parent-child relationships between classes
    mask_mat_t2 = get_mask_matrix(taxonomy['tier1'], class_to_idx['tier1'], class_to_idx['tier2'])
    mask_mat_t3 = get_mask_matrix(taxonomy['tier1'], class_to_idx['tier2'], class_to_idx['tier3'])

    # Instantiate the model and tokenizer
    model_name = "FacebookAI/roberta-base"
    print(f'Instantiating model from {model_name}...')
    model = HierarchicalBERT.from_pretrained(model_name,
                                             num_labels = len(classes['tier1']),
                                             num_tier2 = len(classes['tier2']),
                                             num_tier3 = len(classes['tier3']),
                                             mask_tier2 = mask_mat_t2,
                                             mask_tier3 = mask_mat_t3)
    print(f'Instantiating tokenizer from {model_name}...')
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Converting classes to integers
    y1_train = [class_to_idx[c] for c in train_set['tier1']]
    y2_train = [class_to_idx[c] for c in train_set['tier2']]
    y3_train = [class_to_idx[c] for c in train_set['tier3']]
    y1_eval = [class_to_idx[c] for c in eval_set['tier1']]
    y2_eval = [class_to_idx[c] for c in eval_set['tier2']]
    y3_eval = [class_to_idx[c] for c in eval_set['tier3']]

    # Instantiate the datasets
    print('Tokenizing data...')
    train_dataset = Tier3Dataset(train_set['description'], y1_train, y2_train, y3_train, tokenizer)
    eval_dataset = Tier3Dataset(eval_set['description'], y1_eval, y2_eval, y3_eval, tokenizer)
    print('Tokenization finished')

    # Setting up the trainer
    data_collator = HierarchicalDataCollator(tokenizer)
    training_args = TrainingArguments(output_dir = 'roberta-t1-genres',
                                      eval_strategy = 'epoch',
                                      save_strategy = 'epoch')
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator= data_collator,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        compute_metrics = compute_metrics
        )
    
    trainer.train()

    model.config.id2label = idx_to_class['tier1']
    model.config.label2id = class_to_idx['tier1']

    return model, tokenizer

def main():
    download_dataset()

    train_set = parse_dataset('/content/data/BlurbGenreCollection_EN_train.txt')
    eval_set = parse_dataset('/content/data/BlurbGenreCollection_EN_dev.txt')

    model, tokenizer = train(train_set, eval_set)

    save_model('models', model, tokenizer)

if __name__=="__main__":
    main()
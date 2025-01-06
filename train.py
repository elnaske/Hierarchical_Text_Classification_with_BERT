from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel, TrainingArguments, Trainer
import torch
from torch import nn

from dataset import download_dataset, parse_dataset, get_classes, Tier1Dataset
from trainer_utils import compute_metrics_t1, save_model

def train_t1(train_set, eval_set):
    classes, class_to_idx, idx_to_class, taxonomy = get_classes(train_set)

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

def main():
    download_dataset()

    train_set = parse_dataset('/content/data/BlurbGenreCollection_EN_train.txt')
    eval_set = parse_dataset('/content/data/BlurbGenreCollection_EN_dev.txt')

    model, tokenizer = train_t1(train_set, eval_set)

    save_model('models', model, tokenizer)

if __name__=="__main__":
    main()
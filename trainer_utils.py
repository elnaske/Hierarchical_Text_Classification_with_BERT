import os
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

def save_model(save_name, model, tokenizer):
    dir = 'models'
    save_path = os.path.join(dir, save_name)
    model_save_path = os.path.join(save_path, "model")
    tokenizer_save_path = os.path.join(save_path, "tokenizer")

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

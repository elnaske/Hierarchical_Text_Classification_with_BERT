import requests
import os
import re
from zipfile import ZipFile
from torch.utils.data import Dataset
import pandas as pd


def download_dataset():
    """
    Donwloads the English Blurb Genre Collection dataset.
    """
    url = 'https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip'
    save_path = url.split('/')[-1]

    if not os.path.exists(save_path):
        print('Downloading data...')
        with requests.get(url) as r:
            with open(save_path, 'wb') as f:
                f.write(r.content)

        with ZipFile(save_path, 'r') as zipfile:
            zipfile.extractall(path = 'data')

        print('Download successful')
    
    else:
        print('Data already exists, skipping download...')

def parse_dataset(file_path):
    """
    Parses the XML for the dataset files and returns a dictionary containing the title, description, and categories for each book.
    """
    # Read the file
    with open(file_path, 'r') as f:
        raw_text = f.read()

    # Define regular expressions for matching the relevant content
    tags = ['title', 'body', 'topics', 'd0', 'd1', 'd2']
    regex = {tag: f'<{tag}>\n?(.*?)\n?</{tag}>' for tag in tags}

    titles = re.findall(regex['title'], raw_text)
    descriptions = re.findall(regex['body'], raw_text)
    topics = re.findall(regex['topics'], raw_text)

    tier1 = []
    tier2 = []
    tier3 = []

    for n in topics:
        # Separate the categories for the different hierarchy levels
        tiers = [re.findall(regex[f'd{i}'], n)[0] if re.findall(regex[f'd{i}'], n) else 'None' for i in range(3)]
        tier1.append(tiers[0])
        tier2.append(tiers[1])
        tier3.append(tiers[2])

    out = {
        'title': titles,
        'description': descriptions,
        'tier1': tier1,
        'tier2': tier2,
        'tier3': tier3
    }

    return out

def get_classes(dataset):
    df = pd.DataFrame(dataset)

    t1_classes = list(df['tier1'].unique())
    t2_classes = list(df['tier2'].unique())
    t3_classes = list(df['tier3'].unique())

    classes = {
        'tier1': t1_classes,
        'tier2': t2_classes,
        'tier3': t3_classes
    }

    # Dictionaries for converting classes to integers and vice versa
    class_to_idx = {'tier1': {c: i for i, c in enumerate(t1_classes)},
                    'tier2': {c: i for i, c in enumerate(t2_classes)},
                    'tier3': {c: i for i, c in enumerate(t3_classes)}}
    idx_to_class = {'tier1': {i: c for c, i in class_to_idx['tier1'].items()},
                    'tier2': {i: c for c, i in class_to_idx['tier2'].items()},
                    'tier3': {i: c for c, i in class_to_idx['tier3'].items()}}
    
    # Parse the hierarchy to get the parent-child relationships
    t1_children = {parent: list(df[(df['tier1'] == parent) & (df['tier2'] != 'None')]['tier2'].unique())
                   for parent in t1_classes}
    t2_children = {parent: list(df[(df['tier2'] == parent) & (df['tier3'] != 'None')]['tier3'].unique())
                   for parent in t2_classes
                   if parent != 'None'}
    taxonomy = {
        'tier1': t1_children,
        'tier2': t2_children
    }
    
    return classes, class_to_idx, idx_to_class, taxonomy


class Tier1Dataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer):
        self.descriptions = descriptions
        self.labels = labels

        self.encoding = tokenizer(
            self.descriptions,
            max_length = 512,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encoding['input_ids'][idx],
            'attention_mask': self.encoding['attention_mask'][idx],
            'labels': self.labels[idx]
        }
    
class Tier3Dataset(Dataset):
    def __init__(self, descriptions, tier1, tier2, tier3, tokenizer):
        self.descriptions = descriptions
        self.tier1 = tier1
        self.tier2 = tier2
        self.tier3 = tier3

        self.encoding = tokenizer(
            self.descriptions,
            max_length = 512,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encoding['input_ids'][idx],
            'attention_mask': self.encoding['attention_mask'][idx],
            't1_labels': self.tier1[idx],
            't2_labels': self.tier2[idx],
            't3_labels': self.tier3[idx]
        }
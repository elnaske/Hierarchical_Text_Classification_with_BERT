# Hierarchical Text Classification with BERT

An approach to hierarchical text classification using BERT-based models. It explicitly limits the classes that can be predicted for lower tiers by masking the logit outputs of the prediction layer with a binary vector designating the dependencies between different levels of the hierarchy.

Based on a project for the course L665 - Applying Machine Learning Techniques in Computational Linguistics at Indiana University Bloomington.
Because the original implementation used a dataset that is not yet publicly availaible, this model was trained on the 2023 version of the Amazon Reviews dataset by Hou et al. 2024.

## Architecture

## Dataset
The model input consisted of the
The model was trained on product meta data for the software category from the [2023 Amazon Reviews dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
- subcategories
- first two levels
- Cleaning (Removed classes with low membership,

## Methodology
- RoBERTa 
- Fine-tune t1
- Train t2

- Hugging Face Trainer API 

## Results

## References


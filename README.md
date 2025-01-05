# Hierarchical Text Classification with BERT

An approach to hierarchical text classification using BERT-based models. It explicitly limits the classes that can be predicted for lower tiers by masking the logit outputs of the prediction layer with a binary vector designating the dependencies between different levels of the hierarchy.

Based on a project for the course L665 - Applying Machine Learning Techniques in Computational Linguistics at Indiana University Bloomington.
Because the original implementation used a dataset that is not yet publicly availaible, this model was trained on the [Blurb Genre Collection](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html) dataset.

## Architecture

## Dataset
The model input consisted of the
- subcategories
- first three levels

## Methodology
- RoBERTa 
- Fine-tune t1
- Train t2

- Hugging Face Trainer API 

## Results


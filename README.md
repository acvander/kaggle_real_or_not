# About
This is my solution for the [Kaggle Real or Not? NLP with disaster tweets](https://www.kaggle.com/c/nlp-getting-started) getting started competition.

# Models
## RNN Model
## BERT Model

# Guide
## Dependency Installation
```pip install -r requirements.txt```

## Running program
All commands are executed using 

```python main.py```

 and using the flags

## Preprocessing data
Preprocessing carries out some cleanup of the data. Run using:

```python main.py --mode=preprocess```

## Training standard models

## Training BERT models
First run the BERT preprocessing

```python main.py --mode=preprocess_bert```

This will generate the training data in a python shelve file created at ```./tmp/bert_data```.

Next run the training program

```python main.py --mode=train_bert```




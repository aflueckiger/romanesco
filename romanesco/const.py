#!/usr/bin/env python3

EOS = '<eos>'
UNK = '<unk>'

MODEL_FILENAME = 'model'
VOCAB_FILENAME = 'vocab.json'

# Ugly hardcoded hyperparameters
NUM_STEPS = 35 # truncated backprop length
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 700 # layer size
EMBEDDING_SIZE = 300 # dimensions of word embeddings
# NUM_LAYERS = 2 # number of hidden layers

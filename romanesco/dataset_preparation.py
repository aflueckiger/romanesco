#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Alex Flückiger

import sys
import re
import glob
import os
from collections import Counter
import spacy


nlp = spacy.load('en', disable=['ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def get_txt_files_in_dir(dir):
    """
    Find relevant files of corpus
    """
    dir = os.getcwd() + dir
    searchpath = dir + '**/*.txt'
    return [fname for fname in glob.glob(searchpath, recursive=True)]


def iter_files(files):
    """
    Yield all documents.
    A list of seeds may be provided to yield only relevant sentences
    """

    i = 0

    for fname in files:

        i += 1
        if i % 1000 == 0:
            print('{} documents processed'.format(i))

        with open(fname) as f:
            text = f.read()
            # remove paragraph numbers
            text = re.sub(r'[.\n]{1,2} ?[\d]+ ?\.', '.', text)
            # remove all line breaks
            text = re.sub(r'\n(\n?)', ' ', text)
            # remove tabs
            text = re.sub(r' ?\t ?', ' ', text)
            yield text


def create_tokenized_corpus(corpusdir, fname_out='corpus'):
    files = get_txt_files_in_dir(corpusdir)

    print('Processing {} documents...'.format(len(files)))

    with open(fname_out, 'w') as f:
        for text in iter_files(files):
            doc = nlp(text, disable=['parser', 'tagger'])
            for sent in doc.sents:
                sent_tokens = [token.text for token in sent]
                sent_tokenized = ' '.join(sent_tokens).strip()
                # only use sentences with at least 3 tokens
                if len(sent_tokens) > 3:
                    f.write(sent_tokenized + '\n')

    print_corpus_stats(fname_out)


def print_corpus_stats(fname):
    vocab = Counter()
    sent_length = list()

    with open(fname) as f_in:
        for line in f_in:
            tokens = line.split()
            vocab.update(tokens)
            sent_length.append(len(tokens))
            vocab.update(tokens)

    print('\nDescriptive statistics:')
    print('Total sentences:', len(sent_length))
    print('Total tokens:', sum(sent_length))
    print('Unique tokens:', len(vocab))
    print('Avg. tokens per sentence:', round(
        sum(sent_length) / len(sent_length), 2))
    print('-'*10, '\n')


def truecase_corpus(fname_in, fname_out=None):
    if fname_out is None:
        fname_out = fname_in + '.truecased'

    vocab = Counter()

    print('Original corpus:')
    print_corpus_stats(fname_in)

    # count all tokens in original corpus
    with open(fname_in) as f_in:
        for line in f_in:
            tokens = line.split()
            vocab.update(tokens)

    # truecase sentences and write into new file
    with open(fname_in) as f_in, open(fname_out, 'w') as f_out:
        for line in f_in:
            tokens = line.split()
            # truecase first token per sentence, cases of other tokens are preserved
            if vocab[tokens[0]] < vocab[tokens[0].lower()]:
                tokens[0] = tokens[0].lower()
            # write truecase sentence into new file
            f_out.write(' '.join(tokens) + '\n')

    print('Truecased corpus:')
    print_corpus_stats(fname_out)


if __name__ == '__main__':

    corpusdir = sys.argv[1]

    create_tokenized_corpus(corpusdir)
    truecase_corpus('corpus')

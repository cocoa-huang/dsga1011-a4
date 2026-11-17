import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

P_CORRUPT = 0.3
ELIGIBLE_TAGS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}                                  # (1)

ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBORS = {}
for r, row in enumerate(ROWS):
    for c, ch in enumerate(row):
        cand = [(r, c - 1), (r, c + 1), (r - 1, c), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c)]
        NEIGHBORS[ch] = [ROWS[rr][cc] for rr, cc in cand if 0 <= rr < 3 and 0 <= cc < len(ROWS[rr])]

def substitute(word):
    i = random.randrange(1, len(word) - 1)                                           
    new = random.choice(NEIGHBORS[word[i].lower()])               
    if word[i].isupper(): new = new.upper()                          
    return word[:i] + new + word[i + 1:]

def transpose(word):
    pairs = [i for i in range(1, len(word) - 2) if word[i] != word[i + 1]]             
    if not pairs:
        return substitute(word)                                    
    i = random.choice(pairs)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:] # swap i and i+1


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    tokens = word_tokenize(example["text"])
    out = []
    for tok, tag in pos_tag(tokens):                       
        if (tag in ELIGIBLE_TAGS and tok.isascii() and tok.isalpha() and len(tok) >= 3
                and random.random() < P_CORRUPT):         
            tok = substitute(tok) if random.random() < 0.5 else transpose(tok) 
        out.append(tok)
    example["text"] = TreebankWordDetokenizer().detokenize(out)


    ##### YOUR CODE ENDS HERE ######

    return example

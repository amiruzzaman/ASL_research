from collections import defaultdict
from typing import Counter
import torch
import numpy
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import spacy

class ASLDataset(Dataset):
    def __init__(self, gloss_set, text_set, gloss_vocab, text_vocab):
        self.gloss_set = gloss_set
        self.text_set = text_set

        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab    

        self.nlp = spacy.blank("en")

    def __len__(self):
        return len(self.gloss_set)
    
    def __getitem__(self, index):
        gloss = self.gloss_set[index]
        text = self.text_set[index]

        # Split sentence of ASL gloss and English text to a list of words (Adds the SOS and EOS also)
        gloss_words = ["<sos>"] + [token.text for token in self.nlp(gloss)] + ["<eos>"]
        text_words = ["<sos>"] + [token.text for token in self.nlp(text)] + ["<eos>"]
        
        # Convert those list of words to tokens/indices in the vocab
        gloss_tokens = torch.tensor([self.gloss_vocab[word] for word in gloss_words])
        text_tokens = torch.tensor([self.text_vocab[word] for word in text_words])
        
        return gloss_tokens, text_tokens

def build_vocab(dataset, special = ["<sos>", "<eos>", "<pad>"]):
    nlp = spacy.blank("en")
    count = Counter()

    # Getting the count of all words in the dataset
    for sentence in nlp.pipe(dataset):
        count.update([token.text for token in sentence])

    # Sort the words based on how many times it appears in the dataset (Largest to smallest)
    words = sorted(count.keys(), key=lambda word: count[word], reverse=True)

    # Combine the special words (SOS, EOS, etc) with the words from the dataset
    vocab = special + words

    # Pairs word with their index in the vocab list (and vice versa)
    word_to_id = {word:id for id, word in enumerate(vocab)}
    id_to_word = {id:word for id, word in enumerate(vocab)}

    return word_to_id, id_to_word

# Processes the list of samples in the batch so that all sample sentences are the same length
def collate_fn(batch):
    x, y = zip(*batch)
    x = [torch.tensor(val) for val in x]
    y = [torch.tensor(val) for val in y]
    
    return pad_sequence(x, batch_first=True, padding_value=2), pad_sequence(y, batch_first=True, padding_value=2)

# Loading the English-ASL Gloss Parallel Corpus 2012 Dataset
print("Loading in Dataset...")
aslg_dataset = load_dataset("achrafothman/aslg_pc12", split='train')
gloss_set, text_set = zip(*[(pair["gloss"].strip(), pair["text"].strip()) for pair in aslg_dataset])
 
print("Building the vocab...")
gloss_vocab, gloss_id = build_vocab(gloss_set)
text_vocab, text_id = build_vocab(text_set)

# Creating Custom ASL Dataset
print("Creating custom ASL Dataset and Dataloader...")
dataset = ASLDataset(gloss_set, text_set, gloss_vocab, text_vocab)
dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate_fn, shuffle=True)

# Fetches a sample from the batch and prints the gloss and token
train_glosses, train_text = next(iter(dataloader))
gloss = train_glosses[0]
text = train_text[0] 

print(f"Gloss: {[gloss_id[val.item()] for val in gloss]}")
print(f"Gloss Tokens: {gloss}\n")
print(f"Text: {[text_id[val.item()] for val in text]}")
print(f"Text Tokens: {text}\n\n")

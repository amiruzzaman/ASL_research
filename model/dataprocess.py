from collections import defaultdict
from typing import Counter
import torch
import numpy
from datasets import load_dataset
from datasets import enable_progress_bars
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import spacy
from tqdm import tqdm

class ASLDataset(Dataset):
    def __init__(self, gloss_set, text_set, gloss_vocab, text_vocab, filters):
        """
        Custom dataset class for the ASLG-PC12 dataset

        Parameters:
            gloss_set: A dictionary that maps a token for an ASL word to a string
            text_set: A dictionary that maps a token for an English word to a string

            gloss_vocab: A dictionary that maps words in the gloss vocab to a numerical token
            text_vocab: A dictionary that maps words in the English vocab to a numerical token
        """
        
        self.gloss_set = gloss_set
        self.text_set = text_set

        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab

        self.filters = filters    

        self.nlp = spacy.blank("en")

    def __len__(self):
        return len(self.gloss_set)
    
    def __getitem__(self, index):
        """
        For a sample in the dictionary, we retrieve the random ASL gloss and English sentence in string form
        Then using their respective vocabs, we convert them into a sequence of tokens with an <SOS> token added to
        the start and an <EOS> token added to the end.

        Parameters:
            index: The location of the sample in the batch
        
        Returns:
            Two sequence of tokens for the glosses and sentences with an <SOS> token added to the start 
            and an <EOS> token added to the end.
        """
        gloss = self.gloss_set[index]
        text = self.text_set[index]

        for filter in self.filters:
            gloss = gloss.replace(filter, "")

        # Split sentence of ASL gloss and English text to a list of words (Adds the SOS and EOS also)
        gloss_words = ["<sos>"] + gloss.split() + ["<eos>"]
        text_words = ["<sos>"] + text.split() + ["<eos>"]
        
        # Convert those list of words to tokens/indices in the vocab
        gloss_tokens = torch.tensor([self.gloss_vocab[word] for word in gloss_words])
        text_tokens = torch.tensor([self.text_vocab[word] for word in text_words])
        
        return gloss_tokens, text_tokens

def build_vocab(dataset, special = ["<sos>", "<eos>", "<pad>", "<unk>"], filters = []):
    """
    Builds the vocabulary for a dataset by splitting each sentence by their words and then
    Create two dictionaries, id_to_word and word_to_id.

    Parameters:
        dataset: The data that is processed into a vocabulary dictionary
        special: Special tokens that are added to the beginning of the vocabulary

    Returns:
        A dictionary that maps each word in the dataset to their token id and another dictionary
        that does the opposite
    """
    count = Counter()

    # Getting the count of all words in the dataset
    for sentence in tqdm(dataset):
        for filter in filters:
            sentence = sentence.replace(filter, "")
        
        count.update(sentence.split())

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
    """
    Processes the list of samples in the batch so that all sample sentences are the same length.

    Parameter: 
        batch: A batch in the dataloader
    
    Returns:
        The batch with both sequences padded 
    """
    x, y = zip(*batch)
    x = [torch.tensor(val) for val in x]
    y = [torch.tensor(val) for val in y]
    
    return pad_sequence(x, batch_first=True, padding_value=2), pad_sequence(y, batch_first=True, padding_value=2)


def get_data(batch_size):
    """
    Loads the ASLG-PC12 Dataset and creates a Dataloader for it.

    Parameters:
        batch_size: How many items each batch will contain

    Returns:
        A dataloader that contains the ASL glosses and the English sentences
    """
    enable_progress_bars()
    filters = ["DESC-", "X-"]
    
    # Loading the English-ASL Gloss Parallel Corpus 2012 Dataset
    print("Loading in Dataset...")
    aslg_dataset = load_dataset("achrafothman/aslg_pc12", split='train')
    gloss_set, text_set = zip(*[(pair["gloss"].strip(), pair["text"].strip()) for pair in aslg_dataset])
    
    print("Building the vocab...")
    gloss_vocab, gloss_id = build_vocab(gloss_set, filters=filters)
    text_vocab, text_id = build_vocab(text_set)

    # Creating Custom ASL Dataset
    print("Creating custom ASL Dataset and Dataloader...\n")
    dataset = ASLDataset(gloss_set, text_set, gloss_vocab, text_vocab, filters)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    
    train_features, train_labels = next(iter(dataloader))
    gloss = train_features[0]
    text = train_labels[0]

    print(gloss)
    print([gloss_id[word] for word in gloss.tolist()])
    print()

    return dataloader, gloss_vocab, gloss_id, text_vocab, text_id
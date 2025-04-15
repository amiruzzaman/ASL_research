from collections import defaultdict
from typing import Counter
import torch
import numpy
from datasets import load_dataset
from datasets import enable_progress_bars
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import spacy
from tqdm import tqdm

class ASLDataset(Dataset):
    def __init__(self, src_set, trg_set, src_vocab, trg_vocab, src_filters=[], trg_filters=[]):
        """
        Custom dataset class for the ASLG-PC12 dataset

        Parameters:
            gloss_set: A dictionary that maps a token for an ASL word to a string
            text_set: A dictionary that maps a token for an English word to a string

            gloss_vocab: A dictionary that maps words in the gloss vocab to a numerical token
            text_vocab: A dictionary that maps words in the English vocab to a numerical token

            gloss_filters: A list that contains certain words to get rid of in the glosses
            text_filters: A list that contains certain words to get rid of in the texts
        """
        
        self.src_set = src_set
        self.trg_set = trg_set

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.src_filters = src_filters
        self.trg_filters = trg_filters    


    def __len__(self):
        return len(self.src_set)
    
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
        src = self.src_set[index]
        trg = self.trg_set[index]

        # Gets rid of certain words in the keywords
        for filter in self.src_filters:
            src = src.replace(filter, "")

        for filter in self.trg_filters:
            trg = trg.replace(filter, "")

        # Split sentence of ASL gloss and English text to a list of words (Adds the SOS and EOS also)
        src_words = ["<sos>"] + src.split() + ["<eos>"]
        trg_words = ["<sos>"] + trg.split() + ["<eos>"]
            
        # Convert those list of words to tokens/indices in the vocab
        src_tokens = torch.tensor([self.src_vocab[word] for word in src_words])
        trg_tokens = torch.tensor([self.trg_vocab[word] for word in trg_words])
        
        return src_tokens, trg_tokens

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


def load_alsg_dataset(batch_size=1, random_state=29, test_size=0.1, reverse = False):
    """
    Loads the ASLG-PC12 Dataset and creates a Dataloader for it.
    
    Parameters:
        batch_size: How many items each batch will contain
        random_state: Controls the shuffling applied on the data during splitting.  
        test_size: The proportion of the dataset to include in the test dataset

    Returns:
        A dataloader that contains the ASL glosses and the English sentences
    """
    enable_progress_bars()
    filters = ["desc-", "x-"]

    # Loading the English-ASL Gloss Parallel Corpus 2012 Dataset
    print("Loading in Dataset...")
    aslg_dataset = load_dataset("achrafothman/aslg_pc12", split='train')
    glosses, texts = zip(*[(pair["gloss"].strip(), pair["text"].strip()) for pair in aslg_dataset])

    glosses = list(map(lambda x: x.lower(), glosses))
    texts = list(map(lambda x: x.lower(), texts))

    print("Building the vocab...")
    gloss_vocab, gloss_id = build_vocab(glosses, filters=filters)
    text_vocab, text_id = build_vocab(texts)

    # Split data into a training and validation set
    gloss_train, gloss_test, english_train, english_test = train_test_split(glosses, texts, random_state=29, test_size=test_size, shuffle=True)
        
    # Creating Custom ASL Dataset
    print("Creating custom ASL Dataset and Dataloader...\n")
    train_dataset, test_dataset = None, None

    if not reverse:
        train_dataset = ASLDataset(gloss_train, english_train, gloss_vocab, text_vocab, src_filters=filters)
        test_dataset = ASLDataset(gloss_test, english_test, gloss_vocab, text_vocab, src_filters=filters)
    else:
        train_dataset = ASLDataset(english_train, gloss_train, text_vocab, gloss_vocab, trg_filters=filters)
        test_dataset = ASLDataset(english_test, gloss_test, text_vocab, gloss_vocab, trg_filters=filters)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    return train_dl, test_dl, gloss_vocab, gloss_id, text_vocab, text_id
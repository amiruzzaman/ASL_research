from typing import Counter
import torch
from datasets import load_dataset
from datasets import enable_progress_bars
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd

from ml.models.asl_to_english_v1.vocab import Vocabulary

DATASET_PATH = os.path.join("src", "ml", "data", "processed", "aslgpc12", "dataset.csv")


class TextDataset(Dataset):
    def __init__(
        self, src: list, trg: list, src_vocab: Vocabulary, trg_vocab: Vocabulary
    ):
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

        self.src = src
        self.trg = trg

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src)

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

        src_sample = self.src[index]
        trg_sample = self.trg[index]

        # Split sentence of ASL gloss and English text to a list of words (Adds the SOS and EOS also)
        src_tokens = self.src_vocab.tokenize(src_sample)
        trg_tokens = self.trg_vocab.tokenize(trg_sample)

        return src_tokens, trg_tokens


def build_vocab(dataset, special=["<sos>", "<eos>", "<pad>", "<unk>"]):
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
        count.update(sentence.split())

    # Sort the words based on how many times it appears in the dataset (Largest to smallest)
    words = sorted(count.keys(), key=lambda word: count[word], reverse=True)

    # Combine the special words (SOS, EOS, etc) with the words from the dataset
    vocab = special + words

    # Pairs word with their index in the vocab list (and vice versa)
    word_to_id = {word: id for id, word in enumerate(vocab)}
    id_to_word = {id: word for id, word in enumerate(vocab)}

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

    return pad_sequence(x, batch_first=True, padding_value=2), pad_sequence(
        y, batch_first=True, padding_value=2
    )


def load_alsg_dataset(batch_size=1, random_state=29, test_size=0.3, reverse=False):
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

    # Loading the English-ASL Gloss Parallel Corpus 2012 Dataset
    print("Loading in Dataset...")
    df = pd.read_csv(DATASET_PATH)

    glosses = df["gloss"].tolist()
    texts = df["text"].tolist()

    print("Building the vocab...")
    gloss_vocab, gloss_id = build_vocab(glosses)
    text_vocab, text_id = build_vocab(texts)

    # Split data into a training, validation, and test set
    gloss_train, gloss_test, text_train, text_test = train_test_split(
        glosses, texts, random_state=29, test_size=test_size, shuffle=True
    )

    gloss_valid, gloss_test, text_valid, text_test = train_test_split(
        gloss_test, text_test, random_state=29, test_size=0.5, shuffle=True
    )

    print(len(gloss_train))
    print(len(gloss_valid))
    print(len(gloss_test))

    # Creating Custom ASL Dataset
    print("Creating custom ASL Dataset and Dataloader...\n")
    train_dataset, valid_dataset, test_dataset = None, None, None

    if not reverse:
        train_dataset = ASLDataset(gloss_train, text_train, gloss_vocab, text_vocab)
        valid_dataset = ASLDataset(gloss_valid, text_valid, gloss_vocab, text_vocab)
        test_dataset = ASLDataset(gloss_test, text_test, gloss_vocab, text_vocab)

    else:
        train_dataset = ASLDataset(text_train, gloss_train, text_vocab, gloss_vocab)
        valid_dataset = ASLDataset(text_valid, gloss_valid, text_vocab, gloss_vocab)
        test_dataset = ASLDataset(text_test, gloss_test, text_vocab, gloss_vocab)

    train_dl = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    valid_dl = DataLoader(
        valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    test_dl = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    return train_dl, valid_dl, test_dl, gloss_vocab, gloss_id, text_vocab, text_id

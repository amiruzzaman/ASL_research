from typing import Counter
import torch
from datasets import load_dataset
from datasets import enable_progress_bars
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
import unicodedata

# Creating Dataframe
aslg_dataset = load_dataset("achrafothman/aslg_pc12", split="train")
glosses, texts = zip(
    *[(pair["gloss"].strip(), pair["text"].strip()) for pair in aslg_dataset]
)

df = pd.DataFrame({"glosses": glosses, "texts": texts})

# Stripping columns
df["texts"] = df["texts"].apply(lambda s: s.strip())
df["glosses"] = df["glosses"].apply(lambda s: s.strip())

# Get rid of FEFF unicode character
df["glosses"] = df["glosses"].str.replace(u'\uFEFF', '')

# Dropping duplicates and n/a rows
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset='glosses', keep="last", inplace=True)
df.drop_duplicates(subset='texts', keep="last", inplace=True)
df.dropna()

# Removing . or ? or ! from glosses and texts column
df["glosses"] = df["glosses"].str.replace(r'[.?!]+', '', regex=True)
df["texts"] = df["texts"].str.replace(r'[.?!]+', '', regex=True)

# Removing samples with ()
df = df[~df["glosses"].str.contains(r"\(.+\)")]

# Removing special tags from samples
df["glosses"] = df["glosses"].str.replace(r'(DESC-|X-)+', '', regex=True)

# Removing any rows with numbers in them
df = df[~df["glosses"].str.contains(r'[0-9]+')]

# Replace accented characters with non accented counterparts
df["texts"] = df["texts"].apply(lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)))

print(df)
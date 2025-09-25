from datasets import load_dataset
from datasets import enable_progress_bars
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
import unicodedata
import os

# Save path
SAVE_PATH = os.path.join("src", "ml", "data", "processed", "aslgpc12")

# Creating Dataframe
aslg_dataset = load_dataset("achrafothman/aslg_pc12", split="train")
glosses, texts = zip(
    *[(pair["gloss"].strip(), pair["text"].strip()) for pair in aslg_dataset]
)

df = pd.DataFrame({"glosses": glosses, "texts": texts})

# Stripping columns
df["texts"] = df["texts"].apply(lambda s: s.strip())
df["glosses"] = df["glosses"].apply(lambda s: s.strip())

# Makes text column fully lowercase and gloss column fully uppercase
df["texts"] = df["texts"].apply(lambda s: s.lower())
df["glosses"] = df["glosses"].apply(lambda s: s.upper())

# Gets rid of FEFF unicode character
df["glosses"] = df["glosses"].str.replace(u'\uFEFF', '')

# Dropping duplicates and n/a rows
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset='glosses', keep="last", inplace=True)
df.drop_duplicates(subset='texts', keep="last", inplace=True)
df.dropna()

# Removing . or ? or ! from glosses and texts column
df["glosses"] = df["glosses"].str.replace(r'[.?!/,`]+', '', regex=True)
df["texts"] = df["texts"].str.replace(r'[.?!,/]+', '', regex=True)

# Removing samples with (any text)
df = df[~df["glosses"].str.contains(r"\(.+\)")]

# Removing special tags from samples
df["glosses"] = df["glosses"].str.replace(r'DESC-RE\s', 'THERE ', regex=True)
df["glosses"] = df["glosses"].str.replace(r'X-Y\s', 'THEY ', regex=True)
df["glosses"] = df["glosses"].str.replace(r'DESC-', '', regex=True)
df["glosses"] = df["glosses"].str.replace(r'X-', '', regex=True)

# Removing any rows with numbers in them
df = df[~df["glosses"].str.contains(r'\d+')]

# Remove poss
df["texts"] = df["texts"].str.replace(r'poss', '', regex=True)
df["glosses"] = df["glosses"].str.replace(r'POSS', '', regex=True)

# Replace accented characters with non accented counterparts
df["texts"] = df["texts"].apply(lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)))

df["glosses"] = df["glosses"].str.replace(r"\s\s+", " ", regex=True)
df["texts"] = df["texts"].str.replace(r"\s\s+", " ", regex=True)

df["texts"] = df["texts"].apply(lambda s: s.strip())
df["glosses"] = df["glosses"].apply(lambda s: s.strip())

# Save dataset
df.to_csv(os.path.join(SAVE_PATH, "dataset.csv"), index=False)
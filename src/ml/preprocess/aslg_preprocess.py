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

def preprocess():
    # Creating Dataframe
    aslg_dataset = load_dataset("achrafothman/aslg_pc12", split="train")
    glosses, texts = zip(
        *[(pair["gloss"].strip(), pair["text"].strip()) for pair in aslg_dataset]
    )
    
    df = pd.DataFrame({"gloss": glosses, "text": texts})
    
    # Stripping columns
    df["text"] = df["text"].apply(lambda s: s.strip())
    df["gloss"] = df["gloss"].apply(lambda s: s.strip())
    
    # Makes text column fully lowercase and gloss column fully uppercase
    df["text"] = df["text"].apply(lambda s: s.lower())
    df["gloss"] = df["gloss"].apply(lambda s: s.upper())
    
    # Gets rid of FEFF unicode character
    df["gloss"] = df["gloss"].str.replace(u'\uFEFF', '')
    
    # Dropping duplicates and n/a rows
    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset='gloss', keep="last", inplace=True)
    df.drop_duplicates(subset='text', keep="last", inplace=True)
    df.dropna()
    
    # Removing . or ? or ! from glosses and texts column
    df["gloss"] = df["gloss"].str.replace(r'[.?!/,`]+', ' ', regex=True)
    df["text"] = df["text"].str.replace(r'[.?!,/]+', ' ', regex=True)
    
    # Removing samples with (any text)
    df = df[~df["gloss"].str.contains(r"\(.+\)")]
    
    # Removing special tags from samples
    df["gloss"] = df["gloss"].str.replace(r'DESC-RE[^\s]*\s', 'THERE ', regex=True)
    df["gloss"] = df["gloss"].str.replace(r'X-Y\s', 'THEY ', regex=True)
    df["gloss"] = df["gloss"].str.replace(r'DESC-', '', regex=True)
    df["gloss"] = df["gloss"].str.replace(r'X-', '', regex=True)
    
    # Removing any rows with numbers in them
    df = df[~df["gloss"].str.contains(r'\d+')]
    
    # Remove poss
    df["text"] = df["text"].str.replace(r'poss', '', regex=True)
    df["gloss"] = df["gloss"].str.replace(r'POSS', '', regex=True)
    
    # Replace accented characters with non accented counterparts\\
    df["gloss"] = df["gloss"].str.replace(r"\s\s+", " ", regex=True)
    df["text"] = df["text"].str.replace(r"\s\s+", " ", regex=True)
    
    df["text"] = df["text"].apply(lambda s: s.strip())
    df["gloss"] = df["gloss"].apply(lambda s: s.strip())
    
    # Save dataset
    df.to_csv(os.path.join(SAVE_PATH, "dataset.csv"), index=False)

if __name__ == "__main__":
    preprocess()
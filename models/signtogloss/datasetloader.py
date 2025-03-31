import json
import os
import msgpack
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

DATA_PATH = os.path.join("data", "signs")
class SignToVideoDataset(Dataset):
    def __init__(self, glosses, features):
        super().__init__()
        self.glosses = glosses
        self.features = features
        
    def __len__(self):
        return len(self.glosses)
    
    def __getitem__(self, index):
        return self.glosses[index], self.features[index]


def load_data(batch_size=1, random_state=29, test_size=0.1):
    labels = list(filter(lambda file: os.path.isdir(os.path.join(DATA_PATH, file)), os.listdir(DATA_PATH)))
    glosses = []
    features = []

    for label in labels:
        folder = os.path.join(DATA_PATH, label)
        for sample in os.listdir(folder):
            file_path = os.path.join(folder, sample)
                
            if not os.path.isfile(file_path):
                continue

            with open(file_path, 'r') as file:
                data = json.load(file)
            
            glosses.append(label)
            features.append(torch.tensor(data))

    gloss_train, gloss_test, features_train, features_test = \
    train_test_split(glosses, features, random_state=29, test_size=test_size, shuffle=True)

    train = SignToVideoDataset(gloss_train, features_train)
    test = SignToVideoDataset(gloss_test, features_test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    id_to_gloss = {id:gloss for id, gloss in enumerate(labels)}
    gloss_to_id = {gloss:id for id, gloss in enumerate(labels)}

    return len(labels), train_loader, test_loader, id_to_gloss, gloss_to_id


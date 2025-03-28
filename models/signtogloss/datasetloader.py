import msgpack
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class SignToVideoDataset(Dataset):
    def __init__(self, glosses, features):
        super().__init__()
        self.glosses = glosses
        self.features = features
        
    def __len__(self):
        return len(self.glosses)
    
    def __getitem__(self, index):
        return self.glosses[index], torch.tensor(self.features[index])

def collate_fn(batch):
    glosses, features = zip(*batch)
    seq_len = [len(feature) for feature in features]
    
    padded = pad_sequence(features, batch_first=True)
    return glosses, padded, seq_len

def load_data(batch_size, random_state=29, test_size=0.1):
    with open('wlasl.json', 'rb') as file:
        byte_data = file.read()

    wlasl = msgpack.unpackb(byte_data)  
    data = [(sample["label"], sample["features"]) for sample in wlasl["samples"]]
    glosses, features = zip(*data)
    
    dataset = SignToVideoDataset(glosses, features)

    gloss_train, gloss_test, features_train, features_test = \
    train_test_split(glosses, features, random_state=29, test_size=test_size, shuffle=True)

    train = SignToVideoDataset(gloss_train, features_train)
    test = SignToVideoDataset(gloss_test, features_test)

    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    return wlasl["num_classes"], wlasl["classes"], {gloss:id for id, gloss in enumerate(wlasl["classes"])}, \
        train_loader, test_loader


# out = torch.tensor([
#     [[9, 1], [0, 2], [0, 3]],
#     [[0, 1], [0, 12], [0, 3]],
#     [[0, 1], [0, 2], [0, 3]],
# ])

# test = torch.tensor([[1, 2, 3],
#         [1, 5, 3],
#         [1, 2, 3]])

# seq_len = [0, 1, 0]
# print(torch.cat([out[i, length, :] for i, length in enumerate(seq_len)]).reshape(-1, out.shape[-1]))
# print(out[range(out.shape[0]), seq_len, :])    


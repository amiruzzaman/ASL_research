import torch
import torch.nn as nn
from positionalencoding import PositionalEncoding

class Translator(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, heads, num_encoders=1, num_decoders=1, 
                 dropout=0.1, max_len=1000, activation="relu"):
        super(Translator, self).__init__()

        # Embedding layers for the source and target inputs
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # Positional encodings for the embedding vectors
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = nn.Transformer(d_model, heads, num_encoders, num_decoders)
        
        self.linear = nn.Linear(d_model, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, src_mask, trg_mask, src_padding_mask, trg_padding_mask, memory_padding_mask):
        """
        Feeds the sequences of ASL gloss tokens with their accompanying English translation
        into the translation model to retrieve a probability distribution of trg_vocal_size possible outcomes.
        
        Parameters:
            src (Tensor): A batch containing sequences of ASL glosses. 
            trg (Tensor): A batch containing sequences of English text.  
            
            src_mask (Tensor): A matrix for the source sequences that masks out future elements during self attention
            trg_mask (Tensor): A matrix for the target sequences that masks out future elements during self attention

            src_padding_mask (Tensor): A matrix for the source sequences that masks out padding tokens during self attention
            trg_padding_mask (Tensor): A matrix for the target sequences that masks out padding tokens elements during self attention

        Returns:
            Tensor: A tensor containing vectors that represents a probability distribution of trg_vocal_size possible outcomes
            in each row.
        """

        # Turn the list of tokens (Batch, Sequence Size) into its embedding vectors
        # Then, apply positional encoding to those vectors (Batch, Sequence Size, Embedding Vector Size)
        src_pos = self.pos_encoding(self.src_embedding(src))
        trg_pos = self.pos_encoding(self.trg_embedding(trg))

        # Feed the source and target embedding matrices into the transformer model
        out = self.transformer(src, trg, src_mask, trg_mask, None, src_padding_mask, trg_padding_mask, trg_padding_mask)

        # Feeds the output of the decoders into a linear function that output a vector of size trg_vocal_size 
        # and then applys the softmax activation function on it to receive a probability distribution for each sequence in the batch
        return self.softmax(self.linear(out))
    

    def encode(self, src, src_mask):
        src_pos = self.pos_encoding(self.src_embedding(src))
        out = self.transformer.encoder(src, src_mask)

        return out
    
    def decode(self, trg, memory, trg_mask):
        trg_pos = self.pos_encoding(self.trg_embedding(trg))
        out = self.transformer.decoder(trg, memory, trg_mask)

        return out
        
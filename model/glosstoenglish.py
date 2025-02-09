import torch
import torch.nn as nn
from positionalencoding import PositionalEncoding

class Translator(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=512, heads=8, num_encoders=1, num_decoders=1, dropout=0.1, max_len=1000, activation="relu"):
        """
        Creates the translator nn module.

        Converts the source and target token sequences (Batch Size, Sequence Size) to their embedding tensors (Batch Size, Sequence Size, d_model).          
        Feeds them through the translator model.

        Parameters:
            src_vocab_size: The size of the source sequence's vocabulary
            trg_vocab_size: The size of the target sequence's vocabulary

            d_model: The dimensions of the embedding vectors
            heads: The total number of heads during multi-headed attention

            num_encoders: The number of encoders
            num_decoders: The number of decoders

            dropout: The probability of nodes that are dropped during training
            max_len: The largest length a sequence can be

            activation: The activation function for the layers in the model        
        """

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
            src: A batch containing sequences of ASL glosses
            trg: A batch containing sequences of English text
            
            src_mask: A matrix for the source sequences that masks out future elements
            trg_mask: A matrix for the target sequences that masks out future elements

            src_padding_mask: A matrix for the source sequences that masks out padding tokens
            trg_padding_mask: A matrix for the target sequences that masks out padding tokens elements

        Returns:
            A tensor (Batch, Sequence size, Target Vocab Size) containing vectors that represents 
            a probability distribution of trg_vocal_size possible outcomes in each row.
        """

        # Turn the list of tokens (Batch, Sequence Size) into its embedding vectors
        # Then, apply positional encoding to those vectors (Batch, Sequence Size, Embedding Vector Size)
        src_pos = self.pos_encoding(self.src_embedding(src))
        trg_pos = self.pos_encoding(self.trg_embedding(trg))

        # Feed the source and target embedding matrices into the transformer model
        out = self.transformer(src, trg, src_mask, trg_mask, None, src_padding_mask, trg_padding_mask, trg_padding_mask, memory_padding_mask)

        # Feeds the output of the decoders into a linear function that output a vector of size trg_vocal_size 
        # and then applys the softmax activation function on it to receive a probability distribution for each sequence in the batch
        return self.softmax(self.linear(out))
    

    def encode(self, src, src_mask):
        """
        Feeds the source sequence along with its mask into the encoder to retrieve the fixed length encoding vector
        used for memory during multiheaded attention in the decoder

        Parameters:
            src: The source sequence (Batch, Sequence Size)
            src_mask: The look ahead mask used to mask out future words during encoding 

        Returns:
            A tensor (Batch, Sequence Size, Number of expected features) 
        """

        src_pos = self.pos_encoding(self.src_embedding(src))
        out = self.transformer.encoder(src, src_mask)

        return out
    
    def decode(self, trg, memory, trg_mask):
        """
        Feeds the target sequence along with its mask and memory created from the encoders

        Parameters:
            trg: The target sequence (Batch, Sequence Size)
            trg_mask: The look ahead mask used to mask out future words during decoder 
        
        Returns:
            A tensor (Batch, Sequence Size, Number of expected features) 
        """

        trg_pos = self.pos_encoding(self.trg_embedding(trg))
        out = self.transformer.decoder(trg, memory, trg_mask)

        return self.softmax(self.linear(out))
        
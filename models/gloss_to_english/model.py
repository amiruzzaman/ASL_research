import math
import sys

import torch
import torch.nn as nn
from models.glosstoenglish.positional_encoding import PositionalEncoding
from models.utils import generate_square_subsequent_mask
from models.utils import create_mask

class GlossToEnglishModel(nn.Module):
    def __init__(self, src_vocab_size=-1, trg_vocab_size=-1, d_model=512, heads=8, num_encoders=1, num_decoders=1, dropout=0.1, max_len=1000, activation="relu", device='cpu'):
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

        super(GlossToEnglishModel, self).__init__()
        if src_vocab_size == -1 or trg_vocab_size == -1:
            return

        # Embedding layers for the source and target inputs
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, device=device).to(device)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model, device=device).to(device)

        # Positional encodings for the embedding vectors
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len, device=device).to(device)

        self.transformer = nn.Transformer(d_model=d_model, nhead=heads, num_encoder_layers=num_encoders, 
            num_decoder_layers=num_decoders, dropout=dropout, activation=activation, batch_first=True).to(device)
        
        self.linear = nn.Linear(d_model, trg_vocab_size).to(device)
        self.softmax = nn.Softmax(dim=-1).to(device)
        self._init_weights()

        
    def forward(self, src, trg, src_mask, trg_mask, src_padding_mask, trg_padding_mask, memory_padding_mask):
        """
        Feeds the sequences of ASL gloss tokens with their accompanying English translation
        into the translation model to retrieve a probability distribution of trg_vocal_size possible outcomes.
        
        Parameters:
            src: A batch containing sequences of ASL glosses (Batch, Sequence)
            trg: A batch containing sequences of English text (Batch, Sequence)
            
            src_mask: A matrix for the source sequences that masks out future elements (Batch, Sequence, Sequence)
            trg_mask: A matrix for the target sequences that masks out future elements (Batch, Sequence, Sequence)

            src_padding_mask: A matrix for the source sequences that masks out padding tokens (Batch, Sequence)
            trg_padding_mask: A matrix for the target sequences that masks out padding tokens elements (Batch, Sequence)

        Returns:
            A tensor (Batch, Sequence size, Target Vocab Size) containing vectors that represents 
            a probability distribution of trg_vocal_size possible outcomes in each row.
        """

        # Turn the list of tokens (Batch, Sequence Size) into its embedding vectors
        # Then, apply positional encoding to those vectors (Batch, Sequence Size, Embedding Vector Size)
        src = self.pos_encoding(self.src_embedding(src)) 
        trg = self.pos_encoding(self.trg_embedding(trg))

        # Feed the source and target embedding matrices into the transformer model
        out = self.transformer(src, trg, src_mask, trg_mask, 
                               src_key_padding_mask=src_padding_mask, 
                               tgt_key_padding_mask=trg_padding_mask, 
                               memory_key_padding_mask=memory_padding_mask)
        
        out = self.linear(out)

        # Feeds the output of the decoders into a linear function that output a vector of size trg_vocal_size 
        # and then applys the softmax activation function on it to receive a probability distribution for each sequence in the batch
        return out

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        out = self.transformer.encoder(src_pos, src_mask)

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
        out = self.transformer.decoder(trg_pos, memory, trg_mask)

        return self.linear(out[:, -1])
    

    def greedy_decode(self, src, src_mask, src_vocab, trg_vocab, device, max_len=100):
        self.eval()

        # Convert the sequences from (Sequence) to (Batch, Sequence)
        src = src.unsqueeze(0).to(device)

        # Feed the source sequence and its mask into the transformer's encoder
        memory = self.encode(src, src_mask)
        
        # Creates the sequence tensor to be feed into the decoder: [["<sos>"]]
        sequence = torch.ones(1, 1).fill_(trg_vocab["<sos>"]).type(torch.long).to(device)

        for _ in range(max_len):
            mask = generate_square_subsequent_mask(sequence.shape[-1], device).type(torch.bool).to(device)
                
            # Feeds the target and retrieves a vector (Batch, Sequence Size, Target Vocab Size)
            out = self.decode(sequence, memory, mask)
            _, next_word = torch.max(out, dim=1)
            next_word = torch.tensor([[next_word.item()]]).to(device=device)

            # Concatenate the predicted token to the output sequence
            sequence = torch.cat((sequence, next_word), dim=1)

            if next_word == trg_vocab["<eos>"]:
                break
                    
        return sequence
    

    def beam_search(self, src, src_mask, src_vocab, trg_vocab, device, max_len=100, beam_size=25, temperature=1.0):
        # Convert the sequences from (Sequence) to (Batch, Sequence)
        src = src.unsqueeze(0).to(device)

        # Feed the source sequence and its mask into the transformer's encoder
        memory = self.encode(src, src_mask)

        # Creates the sequence tensor to be feed into the decoder: [["<sos>"]]
        start = torch.ones(1, 1).fill_(trg_vocab["<sos>"]).type(torch.long).to(device)
        candidates = [(start, 0)]
        
        for _ in range(max_len):
            new_candidates = []
            
            for candidate, score in candidates:
                # We do not want to expand current candidate, if the candidates's sequence reaches <eos>
                if candidate[0, -1].item() == trg_vocab["<eos>"]:
                    continue

                mask = generate_square_subsequent_mask(candidate.shape[-1], device).type(torch.bool).to(device)

                logits = self.decode(candidate, memory, mask)
                # scaled_logits = logits / temperature
                # out = self.softmax(scaled_logits)
                top_k_prob, top_k_idx = torch.topk(logits, beam_size, dim=1)
                
                # For each probability, get the token and its accompanying probability
                for i in range(beam_size):
                    token = top_k_idx[:, i]
                    token_prob = torch.log(top_k_prob[:, i])

                    new_candidate = torch.cat((candidate, torch.tensor([[token]]).to(device)), dim=-1).to(device)
                    new_score = score + token_prob

                    new_candidates.append((new_candidate, new_score))
            
            candidates = sorted(new_candidates, key=lambda candidate: candidate[1], reverse=True)
            candidates = candidates[:beam_size]

            if all(candidate[0, -1].item() == trg_vocab["<eos>"] for candidate, _ in candidates):
                break
        
        sequence, _ = candidates[0]
        return sequence





                    


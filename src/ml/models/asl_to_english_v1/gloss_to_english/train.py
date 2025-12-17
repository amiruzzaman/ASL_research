import os

import time
from ml.dataloaders.aslg_dataloader import load_alsg_dataset
from ml.models.asl_to_english_v1.gloss_to_english.model import TranslatorModel
import warnings
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from ml.utils.transformer import (
    create_mask,
)

import evaluate

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def train_epoch(model, data, optimizer, criterion, src_vocab, trg_vocab, epoch):
    # Set model to training mode
    model.train()
    losses = 0

    # Go through batches in the epoch
    for src, trg in tqdm(data, desc=f"Epoch {epoch}"):
        # Convert source and target inputs into its respective device's tensors (CPU or GPU)
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # Excluding the last element because the last element does not have any tokens to predict
        trg_input = trg[:, :-1]

        # Create the masks for the source and target
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(
            src, trg_input, trg_vocab["<pad>"], DEVICE
        )

        # Feed the inputs through the translation model
        # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model
        # instead of the model's output in the prior timestep
        out = model(
            src,
            trg_input,
            src_mask,
            trg_mask,
            src_padding_mask,
            trg_padding_mask,
            src_padding_mask,
        )

        # For the criterion function to work, we have to concatenate all the batches together for it to work
        # The shape of the tensor will turn from (Batch, Sequence Size, Target Vocab Size)
        # to (Batch * Sequence Size, Target Vocab Size)
        actual = out.reshape(-1, out.shape[-1])
        expected = trg[:, 1:].reshape(-1)

        # We zero the gradients of the model, cadlculate the total loss of the sample
        # Then compute the gradient vector for the model over the loss

        # For the loss function, the reason why the expected is the target sequence offsetted forward by one is
        # because it allows us to compare the next word the model predicts to the actual next word in the sequence
        loss = criterion(actual, expected)
        losses += loss.item()
        loss.backward()

        # Apply the gradient vector on the trainable parameters in the model and reset the gradients
        optimizer.step()
        optimizer.zero_grad()

    losses /= len(data)
    return losses


def validate(model, data, criterion, src_vocab, src_id, trg_vocab, trg_id):
    losses, correct, wrong = 0, 0, 0
    model.eval()
    predicted_sentences = []
    actual_sentences = []
    # Go through batches in the epoch
    for src, trg in tqdm(data, desc="Validating"):
        # Convert source and target inputs into its respective device's tensors (CPU or GPU)
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        # Excluding the last element because the last element does not have any tokens to predict
        trg_input = trg[:, :-1]

        # Create the masks for the source and target
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(
            src, trg_input, trg_vocab["<pad>"], DEVICE
        )
        
        # Feed the inputs through the translation model
        # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model
        # instead of the model's output in the prior timestep
        out = model(
            src,
            trg_input,
            src_mask,
            trg_mask,
            src_padding_mask,
            trg_padding_mask,
            src_padding_mask,
        )
        
        # For the criterion function to work, we have to concatenate all the batches together for it to work
        # The shape of the tensor will turn from (Batch, Sequence Size, Target Vocab Size)
        # to (Batch * Sequence Size, Target Vocab Size)
        actual = out.reshape(-1, out.shape[-1])
        expected = trg[:, 1:].reshape(-1)

        # For the loss function, the reason why the expected is the target sequence offsetted forward by one is
        # because it allows us to compare the next word the model predicts to the actual next word in the sequence
        loss = criterion(actual, expected)
        losses += loss.item()
        
        # Calculate if each sequence in each batch is equal to the expected sequence
        # correct += (out.argmax(dim=2) == trg[:, 1:]).all(dim=1).sum().item()
        # wrong += (out.argmax(dim=2) != trg[:, 1:]).all(dim=1).sum().item()
        
        # Calculate if each word in all batches are equal to the expected sequence
        # correct += (actual.argmax(dim=1) == expected).sum().item()
        # wrong += (actual.argmax(dim=1) != expected).sum().item()
        
        # Translate the series of ASL gloss tokens into a series of English tokens and then convert that series into a string
        predicted = translate(src, model, src_vocab, trg_vocab, trg_id, src_mask, src_padding_mask)
        actual = [decode(tokens, trg_vocab, trg_id) for tokens in trg]
        
        predicted_sentences.extend(predicted)
        actual_sentences.extend(actual)
        
    bleu = bleu_score(predicted=predicted_sentences, actual=actual_sentences)
    rouge1, rougeL = rouge_score(predicted=predicted_sentences, actual=actual_sentences)
    losses /= len(data)
    # correct /= correct + wrong
    
    # print(predicted_sentences)
    # print(actual_sentences)
    # print()
    return losses, correct, bleu, rouge1, rougeL


def translate(sentence, model, src_vocab, trg_vocab, trg_id, src_mask, src_padding_mask):
    num_tokens = sentence.shape[1]
    
    translated_tokens = model.greedy_decode(
        sentence, src_mask, src_padding_mask, src_vocab, trg_vocab, device=DEVICE
    )

    return [decode(tokens, trg_vocab, trg_id) for tokens in translated_tokens]
    
def decode(tokens, trg_vocab, trg_id):
    filter_special_tokens = lambda x: x != trg_vocab["<pad>"] and x != trg_vocab["<sos>"] and x != trg_vocab["<eos>"]
    return (" ".join([trg_id[token] for token in list(filter(filter_special_tokens, tokens.tolist()))]).strip())
    
def bleu_score(predicted, actual):
    if all([len(x) == 0 for x in predicted]):
        return 0

    bleu_results = bleu_metric.compute(predictions=predicted, references=actual)
    return bleu_results['bleu'] * 100
    
def rouge_score(predicted, actual):
    rouge_results = rouge_metric.compute(predictions=predicted, references=actual)
    return rouge_results['rouge1'], rouge_results['rougeL']
    
def train(args):
    train_dl, test_dl, gloss_vocab, gloss_id, text_vocab, text_id = load_alsg_dataset(
        args.batch, reverse=args.reverse
    )
    
    # Creating the translation (Transformer) model
    EPOCHS = args.epochs
    curr_epoch = 1
    model = None
    if not args.reverse:
        model = TranslatorModel(
            len(gloss_vocab),
            len(text_vocab),
            args.dmodel,
            args.heads,
            args.encoders,
            args.decoders,
            device=DEVICE,
        ).to(DEVICE)
    else:
        model = TranslatorModel(
            len(text_vocab),
            len(gloss_vocab),
            args.dmodel,
            args.heads,
            args.encoders,
            args.decoders,
            device=DEVICE,
        ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        ignore_index=text_vocab["<pad>"] if not args.reverse else gloss_vocab["<pad>"]
    ).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=args.adams_ep
    )
    best_loss = torch.inf
    accuracy_history = []
    loss_history = []

    # If the save data argument is not null, then we load the data to continue training
    if args.model_path:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.model_path, weights_only=False)

        curr_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion = checkpoint["criterion"]
        best_loss = checkpoint["best_loss"]
        accuracy_history = checkpoint["accuracy_history"]
        loss_history = checkpoint["loss_history"]

    # Calculate starting performance of the model
    valid_loss, correct, bleu, rouge1, rougeL = (
        validate(model, test_dl, criterion, gloss_vocab, gloss_id, text_vocab, text_id)
        if not args.reverse
        else validate(model, test_dl, criterion, text_vocab, text_id, gloss_vocab, gloss_id)
    )

    print(
        f"Valid Average loss: {valid_loss:>8f}, BLEU Score: {bleu:.2f}, Rouge-1 Score: {rouge1:.2f}, Rouge-L Score: {rougeL:.2f}"
    )

    for epoch in range(curr_epoch, EPOCHS + 1):
        # Train through the entire training dataset and keep track of total time
        start_time = time.time()
        train_loss = (
            train_epoch(
                model, train_dl, optimizer, criterion, gloss_vocab, text_vocab, epoch
            )
            if not args.reverse
            else train_epoch(
                model, train_dl, optimizer, criterion, text_vocab, gloss_vocab, epoch
            )
        )

        # Goes through the validation dataset
        valid_loss, correct, bleu, rouge1, rougeL = (
            validate(model, test_dl, criterion, gloss_vocab, gloss_id, text_vocab, text_id)
            if not args.reverse
            else validate(model, test_dl, criterion, text_vocab, text_id, gloss_vocab, gloss_id)
        )

        loss_history.append(valid_loss)
        accuracy_history.append(loss_history)

        # If the average loss from testing the validation data is smallest than the best model at that point,
        # Then we save the current model
        if valid_loss < best_loss:
            best_loss = valid_loss
            print("New best model, saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "criterion": criterion,
                    "best_loss": best_loss,
                    "loss_history": loss_history,
                    "accuracy_history": accuracy_history,
                    "config": args,
                },
                os.path.join(args.save_path, "best.pt"),
            )

        total_time = time.time() - start_time

        print(f"\nEpoch Time: {total_time:.1f} seconds")
        print(f"Training Average loss: {train_loss:>8f}")
        print(
            f"Valid Average loss: {valid_loss:>8f}, BLEU Score: {bleu:.2f}, Rouge-1 Score: {rouge1:.2f}, Rouge-L Score: {rougeL:.2f}\n"
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ASLGlossModel")

    # Training procedure
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("-b", "--batch", type=int, default=32)
    parser.add_argument("--adams_ep", type=float, default=1e-9)
    parser.add_argument("--factor", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Translation Model Arguments
    parser.add_argument("--dmodel", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoders", type=int, default=2)
    parser.add_argument("--decoders", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--beam_size", type=int, default=25)
    parser.add_argument("--save_path", type=str, default="./")
    args = parser.parse_args()

    # Either train the model or use the model
    train(args)

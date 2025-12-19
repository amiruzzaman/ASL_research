import os

import time
from ml.dataloaders.aslg_dataloader import load_alsg_dataset, load_phoenix_dataset
from ml.models.asl_to_english_v1.gloss_to_english.model import TranslatorModel
import warnings
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import yaml

from ml.models.asl_to_english_v1.vocab import Vocabulary
from ml.utils.transformer import (
    create_mask,
)

import evaluate

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        train_dl,
        valid_dl,
        test_dl,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
    ):
        self.config = config

        self.model_name = self.config.get("model_name", "best")
        self.save_path = self.config.get("save_path", "")
        self.model_path = self.config.get("model_path", None)

        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.curr_epoch = 0
        self.epochs = config.get("epochs", 100)
        self.best_bleu = -torch.inf
        self.accuracy_history = []
        self.loss_history = []

        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.pad_token).to(
            DEVICE
        )

        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        self.wer_metric = evaluate.load("wer")

        self.load()

    def train_epoch(self, epoch):
        # Set model to training mode
        self.model.train()
        losses = 0

        # Go through batches in the epoch
        for src, trg in tqdm(self.train_dl, desc=f"Epoch {epoch}"):
            # Convert source and target inputs into its respective device's tensors (CPU or GPU)
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)

            # Excluding the last element because the last element does not have any tokens to predict
            trg_input = trg[:, :-1]

            # Create the masks for the source and target
            src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(
                src, trg_input, self.trg_vocab.pad_token
            )

            # Feed the inputs through the translation model
            # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model
            # instead of the model's output in the prior timestep
            out = self.model(
                src.to(DEVICE),
                trg_input.to(DEVICE),
                src_mask.to(DEVICE),
                trg_mask.to(DEVICE),
                src_padding_mask.to(DEVICE),
                trg_padding_mask.to(DEVICE),
                src_padding_mask.to(DEVICE),
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
            loss = self.criterion(actual, expected)
            losses += loss.item() * actual.size(0)
            loss.backward()

            # Apply the gradient vector on the trainable parameters in the model and reset the gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

        losses /= len(self.train_dl.dataset)
        return losses

    def validate(self):
        losses = 0
        self.model.eval()
        predicted_sentences = []
        actual_sentences = []
        # Go through batches in the epoch
        for src, trg in tqdm(self.valid_dl, desc="Validating"):
            # Convert source and target inputs into its respective device's tensors (CPU or GPU)
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)

            # Excluding the last element because the last element does not have any tokens to predict
            trg_input = trg[:, :-1]

            # Create the masks for the source and target
            src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(
                src, trg_input, self.trg_vocab.pad_token
            )

            # Feed the inputs through the translation model
            # We are using teacher forcing, a strategy feeds the ground truth or the expected target sequence into the model
            # instead of the model's output in the prior timestep
            out = self.model(
                src.to(DEVICE),
                trg_input.to(DEVICE),
                src_mask.to(DEVICE),
                trg_mask.to(DEVICE),
                src_padding_mask.to(DEVICE),
                trg_padding_mask.to(DEVICE),
                src_padding_mask.to(DEVICE),
            )

            # For the criterion function to work, we have to concatenate all the batches together for it to work
            # The shape of the tensor will turn from (Batch, Sequence Size, Target Vocab Size)
            # to (Batch * Sequence Size, Target Vocab Size)
            actual = out.reshape(-1, out.shape[-1])
            expected = trg[:, 1:].reshape(-1)

            # For the loss function, the reason why the expected is the target sequence offsetted forward by one is
            # because it allows us to compare the next word the model predicts to the actual next word in the sequence
            loss = self.criterion(actual, expected)
            losses += loss.item() * actual.size(0)

            # Translate the series of ASL gloss tokens into a series of English tokens and then convert that series into a string
            predicted = self.translate(src, src_mask, src_padding_mask)
            actual = self.trg_vocab.decode_batch(trg.tolist())

            predicted_sentences.extend(predicted)
            actual_sentences.extend(actual)

        bleu = self.bleu_score(predicted=predicted_sentences, actual=actual_sentences)
        rouge1, rougeL = self.rouge_score(
            predicted=predicted_sentences, actual=actual_sentences
        )
        wer = self.wer_score(predicted=predicted_sentences, actual=actual_sentences)
        losses /= len(self.valid_dl.dataset)

        return {
            "loss": losses,
            "bleu": bleu,
            "rouge1": rouge1,
            "rougeL": rougeL,
            "wer": wer
        }

    def translate(self, sentence, src_mask, src_padding_mask):
        num_tokens = sentence.shape[1]

        translated_tokens = self.model.greedy_decode(
            sentence.to(DEVICE), src_mask.to(DEVICE), src_padding_mask.to(DEVICE)
        )

        return self.trg_vocab.decode_batch(translated_tokens.tolist())

    def train(self):
        # Calculate starting performance of the model
        valid_loss, bleu, rouge1, rougeL = self.validate()

        print(
            f"Valid Average loss: {valid_loss:>8f}, BLEU Score: {bleu:.2f}, Rouge-1 Score: {rouge1:.2f}, Rouge-L Score: {rougeL:.2f}\n"
        )

        for epoch in range(self.curr_epoch, self.epochs + 1):
            # Train through the entire training dataset and keep track of total time
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            print(f"Training Average loss: {train_loss:>8f}\n")

            # Goes through the validation dataset
            valid_loss, bleu, rouge1, rougeL = self.validate()
            self.loss_history.append(valid_loss)
            self.accuracy_history.append(bleu)

            # If the average loss from testing the validation data is smallest than the best model at that point,
            # Then we save the current model
            self.save(epoch, bleu)
            total_time = time.time() - start_time
            print(
                f"Valid Average loss: {valid_loss:>8f}, BLEU Score: {bleu:.2f}, Rouge-1 Score: {rouge1:.2f}, Rouge-L Score: {rougeL:.2f}\n"
            )

    def save(self, epoch, metric):
        if metric < self.best_bleu:
            return

        self.best_bleu = metric
        print("New best model, saving...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "criterion": self.criterion,
                "best_bleu": metric,
                "loss_history": self.loss_history,
                "accuracy_history": self.accuracy_history,
                "config": self.config,
            },
            os.path.join(self.save_path, f"{self.model_name}.pt"),
        )

    def load(self):
        if not self.model_path:
            return

        checkpoint = torch.load(self.model_path, weights_only=False)
        self.curr_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.criterion = checkpoint["criterion"]
        self.best_bleu = checkpoint["best_bleu"]
        self.accuracy_history = checkpoint["accuracy_history"]
        self.loss_history = checkpoint["loss_history"]

    def bleu_score(self, predicted, actual):
        if all([len(x) == 0 for x in predicted]):
            return 0

        bleu_results = self.bleu_metric.compute(
            predictions=predicted, references=actual
        )
        return bleu_results["bleu"] * 100

    def rouge_score(self, predicted, actual):
        rouge_results = self.rouge_metric.compute(
            predictions=predicted, references=actual
        )
        return rouge_results["rouge1"], rouge_results["rougeL"]
    
    def wer_score(self, predicted, actual):
        rouge_results = self.rouge_metric.compute(
            predictions=predicted, references=actual
        )
        return rouge_results["rouge1"], rouge_results["rougeL"]


def create_data(config):
    train_dl, valid_dl, test_dl, gloss_vocab, text_vocab = load_alsg_dataset(
        config.get("batch_size", 32), reverse=config.get("reverse", False)
    )

    return train_dl, valid_dl, test_dl, gloss_vocab, text_vocab


def create_translation_model(
    config: dict, src_vocab: Vocabulary, trg_vocab: Vocabulary
):
    model = TranslatorModel(
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        d_model=config.get("d_model", 512),
        heads=config.get("heads", 8),
        num_encoders=config.get("num_encoders", 2),
        num_decoders=config.get("num_decoders", 2),
        dropout=config.get("dropout", 0.1),
    ).to(DEVICE)

    return model


def main(config):
    train_dl, valid_dl, test_dl, gloss_vocab, text_vocab = create_data(config)
    model = create_translation_model(config, gloss_vocab, text_vocab)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        betas=(0.9, 0.98),
        eps=float(config.get("eps", 1e-9)),
    )

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        src_vocab=gloss_vocab,
        trg_vocab=text_vocab,
    )
    
    trainer.train()

if __name__ == "__main__":
    with open(os.path.join("src", "ml", "configs", "gloss_to_english.yaml")) as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    main(config)

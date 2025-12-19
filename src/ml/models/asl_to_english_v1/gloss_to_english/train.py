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

        self.curr_epoch = 1
        self.epochs = config.get("epochs", 100)
        self.best_metric = -torch.inf
        self.train_history = []
        self.valid_history = []

        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.pad_token, label_smoothing=float(self.config.get("label_smoothing", 0.0))).to(
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

    def validate(self, use_test = False):
        losses = 0
        self.model.eval()
        dl = self.valid_dl if not use_test else self.test_dl
        
        predicted_sentences = []
        actual_sentences = []
        # Go through batches in the epoch
        for src, trg in tqdm(dl, desc="Validating"):
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
            "wer": wer,
        }

    def translate(self, sentence, src_mask, src_padding_mask):
        num_tokens = sentence.shape[1]

        translated_tokens = self.model.greedy_decode(
            sentence.to(DEVICE), src_mask.to(DEVICE), src_padding_mask.to(DEVICE)
        )

        return self.trg_vocab.decode_batch(translated_tokens.tolist())

    def train(self):
        # Calculate starting performance of the model
        metrics = self.validate()
        
        print(
            f"Valid Average loss: {metrics['loss']:>8f}, BLEU Score: {metrics['bleu']:.2f}, Rouge-1 Score: {metrics['rouge1']:.2f}, Rouge-L Score: {metrics['rougeL']:.2f}, WER Score: {metrics['wer']:.2f}%\n"
        )

        for epoch in range(self.curr_epoch, self.epochs + 1):
            # Train through the entire training dataset and keep track of total time
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            print(f"Training Average loss: {train_loss:>8f}\n")

            # Goes through the validation dataset
            metrics = self.validate()
            self.train_history.append(train_loss)
            self.valid_history.append(dict(metrics))
                
            # If the average loss from testing the validation data is smallest than the best model at that point,
            # Then we save the current model
            self.save(epoch, metrics)
            total_time = time.time() - start_time
            print(
                f"Valid Average loss: {metrics['loss']:>8f}, BLEU Score: {metrics['bleu']:.2f}, Rouge-1 Score: {metrics['rouge1']:.2f}, Rouge-L Score: {metrics['rougeL']:.2f}, WER Score: {metrics['wer']:.2f}%\n"
            )
        
        metrics = self.validate(use_test=True)
        print(
            f"Test Average loss: {metrics['loss']:>8f}, BLEU Score: {metrics['bleu']:.2f}, Rouge-1 Score: {metrics['rouge1']:.2f}, Rouge-L Score: {metrics['rougeL']:.2f}, WER Score: {metrics['wer']:.2f}%\n"
        )
    
    def save(self, epoch, metrics):
        if metrics["bleu"] < self.best_metric:
            return
        
        self.best_metric = metrics["bleu"]
        print("New best model, saving...\n")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "train_history": self.train_history,
                "valid_history": self.valid_history,
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
        self.best_metric = checkpoint["best_metric"]
        self.train_history = checkpoint["train_history"]
        self.valid_history = checkpoint["valid_history"]
        self.config = checkpoint["config"]
        
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
        result = self.wer_metric.compute(
            predictions=predicted, references=actual
        )
        return result * 100


def create_data(config):
    train_dl, valid_dl, test_dl, gloss_vocab, text_vocab = None, None, None, None, None

    if config.get("dataset", "aslg-pc12") == "aslg-pc12":
        train_dl, valid_dl, test_dl, gloss_vocab, text_vocab = load_alsg_dataset(
            config.get("batch_size", 32), reverse=config.get("reverse", False)
        )
    else:
        train_dl, valid_dl, test_dl, gloss_vocab, text_vocab = load_phoenix_dataset(
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
        hidden_size=config.get("hidden_size", 2048),
        heads=config.get("heads", 8),
        num_encoders=config.get("num_encoders", 2),
        num_decoders=config.get("num_decoders", 2),
        dropout=config.get("dropout", 0.1),
        norm_first=config.get("norm_first", False),
    ).to(DEVICE)
        
    return model


def main(config):
    train_dl, valid_dl, test_dl, gloss_vocab, text_vocab = create_data(config)
    model = create_translation_model(config, gloss_vocab, text_vocab) if not config.get("reverse", False) else create_translation_model(config, text_vocab, gloss_vocab)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        betas=(0.9, 0.98),
        eps=float(config.get("eps", 1e-9)),
        weight_decay=float(config.get("weight_decay", 0)),
    )
    
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        src_vocab=gloss_vocab if not config.get("reverse", False) else text_vocab,
        trg_vocab=text_vocab if not config.get("reverse", False) else gloss_vocab,
    )
    
    trainer.train()


if __name__ == "__main__":
    with open(os.path.join("src", "ml", "configs", "english_to_gloss.yaml")) as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    main(config)

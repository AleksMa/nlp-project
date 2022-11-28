from config import *
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from transformers import BertTokenizerFast as BertTokenizer, BertModel


class TextDataset(Dataset):
    def __init__(
            self,
            label_clms,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            max_token_len: int = 128,
            test=False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.test = test
        self.label_clms = label_clms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        _id = data_row['id']
        comment_text = data_row.Data[0]

        if not self.test:
            labels = data_row[self.label_clms]

        encoding = self.tokenizer.encode_plus(
            comment_text,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,  # [CLS] & [SEP]
            return_token_type_ids=False,
            return_attention_mask=True,  # attention_mask
            return_tensors='pt',
        )

        if not self.test:
            return dict(
                _id=_id,
                comment_text=comment_text,
                input_ids=encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten(),
                labels=torch.FloatTensor(labels)
            )
        else:
            return dict(
                _id=_id,
                comment_text=comment_text,
                input_ids=encoding["input_ids"].flatten(),
                attention_mask=encoding["attention_mask"].flatten()
            )


class TextTagger(nn.Module):

    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)  # load the pretrained bert model
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)  # add a linear layer to the bert
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output


# function to train the model
def train(train_dataloader, model, device, optimizer, scheduler):
    model.train()

    total_loss, total_accuracy = 0, 0
    avg_loss = 0

    # empty list to save model predictions
    total_preds = []
    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            if DEBUG:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        #     batch = [r.to(device) for r in batch]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # clear previously calculated gradients
        model.zero_grad()
        loss, outputs = model(input_ids, attention_mask, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()
        scheduler.step()

        # model predictions are stored on GPU. So, push it to CPU
        outputs = outputs.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(outputs)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
    if DEBUG:
        print(f"{step}: {avg_loss}")

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate(val_dataloader, model, device):
    if DEBUG:
        print("\nEvaluating...")
    # t0 = time.time()
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            # elapsed = format_time(time.time() - t0)

            # Report progress.
            if DEBUG:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # deactivate autograd
        with torch.no_grad():

            loss, outputs = model(input_ids, attention_mask, labels)

            total_loss = total_loss + loss.item()

            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            total_preds.append(outputs)
            total_labels.append(labels)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)
    if DEBUG:
        print(f"{step}: {avg_loss}")

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    if DEBUG:
        print(f"Evaluate loss {total_loss / len(val_dataloader)}")
    return avg_loss, total_preds, total_labels

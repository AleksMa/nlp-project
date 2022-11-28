from sklearn.metrics import accuracy_score, roc_curve, auc
from text_bert import *
import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast as BertTokenizer, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

pl.seed_everything(RANDOM_SEED)

label_idx = 1
while label_idx < 100:
    df = pd.read_json('data/curdf' + str(label_idx) + '.json')
    #df = pd.read_json('df_top.json')

    df.describe()
    df.head()
    train_df, val_df = train_test_split(df)
    LABEL_COLUMNS = df.columns.tolist()[2:]
    label_idx += DF_IDX_COUNT

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_dataset = TextDataset(
        LABEL_COLUMNS,
        train_df,
        tokenizer,
        max_token_len=MAX_TOKEN_COUNT
    )

    val_dataset = TextDataset(
        LABEL_COLUMNS,
        val_df,
        tokenizer,
        max_token_len=MAX_TOKEN_COUNT
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = TextTagger(len(LABEL_COLUMNS)).to(device)

    model = bert_model

    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * EPOCHS
    warmup_steps = total_training_steps // 5

    optimizer = AdamW(model.parameters(), lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []
    # for each epoch
    for epoch in range(EPOCHS):

        if DEBUG:
            print('\n Epoch {:} / {:}'.format(epoch + 1, EPOCHS))

        # train model
        train_loss, _ = train(train_dataloader, model, device, optimizer, scheduler)

        # evaluate model
        valid_loss, _, _ = evaluate(val_dataloader, model, device)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if DEBUG:
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')


    def evaluate_roc(probs, y_true):

        preds = probs

        y_pred = np.where(preds >= 0.5, 1, 0)
        count_true = 0
        count_pred = 0
        count_acc = 0
        for it, jt in enumerate(y_true):
            if y_true[it] > 0:
                count_true += 1
            if y_pred[it]:
                count_pred += 1
            if y_pred[it] and y_true[it] > 0:
                count_acc += 1
        fpr, tpr, threshold = roc_curve(y_true, preds)
        roc_auc = auc(fpr, tpr)
        print(f'AUC: {roc_auc:.4f}')
        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')


    avg_loss, total_preds, total_labels = evaluate(val_dataloader, model, device)
    for i, name in enumerate(LABEL_COLUMNS):
        print(f"label: {name}")
        evaluate_roc(total_preds[:, i] > 0.5, total_labels[:, i])

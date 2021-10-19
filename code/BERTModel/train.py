import numpy as np
import pandas as pd
import torch

from sklearn import metrics, model_selection
from transformers import AdamW, get_linear_schedule_with_warmup

from config import config
from dataloaders import BERTDataset
from model import BERTBaseUncased
from functions import train_fn, eval_fn

device = torch.device(config.DEVICE)
model = BERTBaseUncased()
model.to(device)

labelled_df = pd.read_csv(
    config.TRAIN_CSV_PATH, 
    encoding='latin-1', 
    usecols = [1, 2], 
    names=["sentiment", "text"]
    )

print(f"size of training dataset: {len(labelled_df)}")

df_train, df_valid = model_selection.train_test_split(
    labelled_df, test_size=0.1, random_state=42, stratify=labelled_df.sentiment.values
)

df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

print(df_train.head())

train_dataset = BERTDataset(
    review=df_train.text.values, target=df_train.sentiment.values
)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle = True, batch_size=config.TRAIN_BATCH_SIZE, num_workers=2
)

valid_dataset = BERTDataset(
    review=df_valid.text.values, target=df_valid.sentiment.values
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
optimizer = AdamW(optimizer_parameters, lr=3e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)

best_accuracy = 0
for epoch in range(config.EPOCHS):
    train_fn(train_data_loader, model, optimizer, device, scheduler)
    outputs, targets = eval_fn(valid_data_loader, model, device)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    print(f"Accuracy Score = {accuracy}")
    if accuracy > best_accuracy:
        torch.save(model.state_dict(), config.MODEL_PATH)
        best_accuracy = accuracy
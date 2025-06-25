# coding=utf-8
# train

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from EnzymeSubstrateModel import EnzymeSubstrateModel
from EnzymeSubstrateDataset import EnzymeSubstrateDataset, collate_fn
import gc

val_dataset = EnzymeSubstrateDataset('data/enzyme_substrate_test_filtereds.xlsx')
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn, num_workers=4, pin_memory=True)


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = EnzymeSubstrateModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = EnzymeSubstrateDataset('data/enzyme_substrate_train_filtereds.xlsx')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4,
                              pin_memory=True)

    for epoch in range(2000):
        model.train()
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch in progress:
            optimizer.zero_grad()

            if batch['enzyme']['esm2'].shape[0] == 0:
                continue
            batch = {
                'enzyme': {
                    'esm2': batch['enzyme']['esm2'].to(device),
                    'structure': batch['enzyme']['structure'].to(device)
                },
                'substrate': {
                    'atom': {
                        k: v.to(device) for k, v in batch['substrate']['atom'].items()
                    },
                    'motif': {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch['substrate']['motif'].items()
                    }
                },
                'label': batch['label'].to(device)
            }
            outputs = model(batch)
            loss = criterion(outputs.squeeze(), batch['label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            del outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
        print(f'Epoch {epoch} Avg Loss: {epoch_loss / len(train_loader):.4f}')
        validate_model(model, device)


def validate_model(model, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            if batch['enzyme']['esm2'].shape[0] == 0:
                continue
            batch = {
                'enzyme': {
                    'esm2': batch['enzyme']['esm2'].to(device),
                    'structure': batch['enzyme']['structure'].to(device)
                },
                'substrate': {
                    'atom': {
                        k: v.to(device) for k, v in batch['substrate']['atom'].items()
                    },
                    'motif': {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch['substrate']['motif'].items()
                    }
                },
                'label': batch['label'].to(device)
            }
            outputs = model(batch)
            preds = (outputs > 0.5).float()
            total_correct += (preds.squeeze() == batch['label']).sum().item()
            del outputs
            torch.cuda.empty_cache()
            gc.collect()
    accuracy = total_correct / len(val_dataset)
    print(f'Validation Accuracy: {accuracy:.2%}')


if __name__ == '__main__':
    train_model()

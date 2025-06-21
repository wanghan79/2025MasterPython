from pandas.core.frame import DataFrame
from torch.utils.data import DataLoader, Dataset
from transformers.file_utils import TensorType
import torch

class VirHostNetData(Dataset):
    def __init__(self, df: DataFrame, tokenizer, max_len, seq_col_a, seq_col_b, target_col, return_mock_data=False):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seq_col_a = seq_col_a
        self.seq_col_b = seq_col_b
        self.target_col = target_col
        self.labeled = self.target_col in df
        self.return_mock_data = return_mock_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq_A = self.df[self.seq_col_a][idx]
        seq_B = self.df[self.seq_col_b][idx]
        
        seq_A = " ".join(seq_A)
        seq_B = " ".join(seq_B)
        
        target = {}
        if self.labeled:
            target = {"labels": torch.as_tensor(self.df[self.target_col][idx], dtype=torch.long)}
        
        if self.return_mock_data:
            return {}, {}, target
        
        tokens_A = self.tokenizer(seq_A, max_length=self.max_len, add_special_tokens=True, padding="max_length",
                            truncation=True, return_tensors=TensorType.PYTORCH)
        tokens_B = self.tokenizer(seq_B, max_length=self.max_len, add_special_tokens=True, padding="max_length",
                            truncation=True, return_tensors=TensorType.PYTORCH)

        # tokens_A['input_ids'].clone().detach()
        tokens_A['input_ids'] = torch.as_tensor(tokens_A['input_ids'], dtype=torch.long).squeeze()
        tokens_A['attention_mask'] = torch.as_tensor(tokens_A['attention_mask'], dtype=torch.long).squeeze()
        # tokens_A['token_type_ids'] = torch.as_tensor(tokens_A['token_type_ids'], dtype=torch.long).squeeze()

        tokens_B['input_ids'] = torch.as_tensor(tokens_B['input_ids'], dtype=torch.long).squeeze()
        tokens_B['attention_mask'] = torch.as_tensor(tokens_B['attention_mask'], dtype=torch.long).squeeze()
        # tokens_B['token_type_ids'] = torch.as_tensor(tokens_B['token_type_ids'], dtype=torch.long).squeeze()

        if self.labeled:
            # return (ids_A, mask_A, ids_B, mask_B, target)
            return tokens_A, tokens_B, target
        else: 
            # return (ids_A, mask_A, ids_B, mask_B)
            return tokens_A, tokens_B, {}

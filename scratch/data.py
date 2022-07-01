import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
sep = tokenizer.sep_token

def preprocess_fn(examples):
    return tokenizer(examples["inputs"], truncation=True)


def get_dds(df, train=True):
    ds = Dataset.from_pandas(df)
    to_remove = ['discourse_text','discourse_type','inputs','discourse_id','essay_id']
    tok_ds = ds.map(preprocess_fn, batched=True, remove_columns=to_remove)
    if train:
        return DatasetDict({"train":tok_ds.select(trn_idxs), "test": tok_ds.select(val_idxs)})
    else: 
        return tok_ds

def split_index(df):
    essay_ids = df.essay_id.unique()
    np.random.seed(42)
    np.random.shuffle(essay_ids)
    essay_ids[:5]

    val_prop = 0.2
    val_sz = int(len(essay_ids)*val_prop)
    val_essay_ids = essay_ids[:val_sz]

    is_val = np.isin(df.essay_id, val_essay_ids)
    idxs = np.arange(len(df))
    val_idxs = idxs[ is_val]
    trn_idxs = idxs[~is_val]

    return val_idxs, trn_idxs

def main():
    df = pd.read_csv("../data/train.csv")
    print(df.head())
    

    df["inputs"] = df.discourse_type + sep + df.discourse_text


    """HuggingFace expects that the target is in a column to be called `label`, and also that the targets are numerical. 
        We will categorize it and create a new column:"""
    new_label = {"discourse_effectiveness": {"Ineffective": 0, "Adequate": 1, "Effective": 2}}
    df = df.replace(new_label)
    df = df.rename(columns = {"discourse_effectiveness": "label"})
    ds = Dataset.from_pandas(df)

    inps = "discourse_text","discourse_type"
    tok_ds = ds.map(preprocess_fn, batched=True, remove_columns=inps+('inputs','discourse_id','essay_id'))

    trn_idxs, val_idxs = split_index(df)

    dds = DatasetDict({"train":tok_ds.select(trn_idxs),
                "test": tok_ds.select(val_idxs)})
    
    return dds

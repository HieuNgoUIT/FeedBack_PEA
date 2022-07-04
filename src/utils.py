import argparse
import os
import random
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn import datasets
from sklearn import model_selection
import pandas as pd

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=False, default=0)
    parser.add_argument("--model", type=str, required=False, default="microsoft/deberta-base")
    parser.add_argument("--lr", type=float, required=False, default=3e-5)
    parser.add_argument("--output", type=str, default=".", required=False)
    parser.add_argument("--input", type=str, default="../input/feedback-prize-effectiveness", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=16, required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--predict", action="store_true", required=False)
    return parser.parse_args()


def _prepare_training_data_helper(args, tokenizer, df, is_train):
    training_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx = row["essay_id"]
        discourse_text = row["discourse_text"]
        discourse_type = row["discourse_type"]

        if is_train:
            filename = os.path.join(args.input, "train", idx + ".txt")
        else:
            filename = os.path.join(args.input, "test", idx + ".txt")

        with open(filename, "r") as f:
            text = f.read()

        encoded_text = tokenizer.encode_plus(
            discourse_type + " " + discourse_text,
            text,
            add_special_tokens=False,
        )
        input_ids = encoded_text["input_ids"]

        sample = {
            "discourse_id": row["discourse_id"],
            "input_ids": input_ids,
            # "discourse_text": discourse_text,
            # "essay_text": text,
            # "mask": encoded_text["attention_mask"],
        }

        if "token_type_ids" in encoded_text:
            sample["token_type_ids"] = encoded_text["token_type_ids"]

        label = row["discourse_effectiveness"]

        sample["label"] = LABEL_MAPPING[label]

        training_samples.append(sample)
    return training_samples


def prepare_training_data(df, tokenizer, args, num_jobs, is_train):
    training_samples = []

    df_splits = np.array_split(df, num_jobs)

    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(args, tokenizer, df, is_train) for df in df_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples


def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # I create a variable so we can stratify on discourse type and effectiveness score at the same time
    data['discourse_type_score'] = data['discourse_type'] + '_' + data['discourse_effectiveness']
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedGroupKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['discourse_type_score'].values, groups=data['essay_id'])):
        data.loc[v_, 'kfold'] = f
    
    # drop the bins column
    data = data.drop("discourse_type_score", axis=1)

    # return dataframe with folds
    return data

if __name__=="__main__":
    # read training data
    df = pd.read_csv("../input/feedback-prize-effectiveness/train.csv")
    df = create_folds(df, num_splits=5)
    #df.kfold.value_counts()
    df.to_csv("train_folds.csv", index=False)
    print('ok')
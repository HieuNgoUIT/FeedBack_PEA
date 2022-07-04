from tqdm import tqdm
import os
import numpy as np
import torch

import pandas as pd

class FeedbackDataset:
    def __init__(self, samples, args, tokenizer):
        self.samples = samples
        self.args = args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]["input_ids"]
        label = self.samples[idx]["label"]

        input_ids = [self.tokenizer.cls_token_id] + ids

        if len(input_ids) > self.args.max_len - 1:
            input_ids = input_ids[: self.args.max_len - 1]

        input_ids = input_ids + [self.tokenizer.sep_token_id]
        mask = [1] * len(input_ids)

        return {
            "ids": input_ids,
            "mask": mask,
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": label,
        }

class Collate:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]
        output["targets"] = [sample["targets"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        output["targets"] = torch.tensor(output["targets"], dtype=torch.long)

        return output


if __name__=="__main__":
    # read training data
    df = pd.read_csv("input/feedback-prize-effectiveness/train.csv")
    df = create_folds(df, num_splits=5)
    #df.kfold.value_counts()
    df.to_csv("train_folds.csv", index=False)
    print('ok')
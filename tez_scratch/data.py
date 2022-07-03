from tqdm import tqdm
import os
import numpy as np
import torch
from joblib import Parallel, delayed

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}

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
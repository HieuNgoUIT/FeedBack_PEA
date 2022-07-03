
import os
import warnings

import numpy as np
import pandas as pd


# from utils import EarlyStopping, prepare_training_data, target_id_map


from torch.nn import functional as F
from transformers import AutoTokenizer

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from utils import *
from data import FeedbackDataset, Collate
from model import FeedbackModel
warnings.filterwarnings("ignore")



def main(args):
    NUM_JOBS = 12
    seed_everything(42)
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(os.path.join(args.input, "train_folds.csv"))

    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    training_samples = prepare_training_data(train_df, tokenizer, args, num_jobs=NUM_JOBS, is_train=True)
    valid_samples = prepare_training_data(valid_df, tokenizer, args, num_jobs=NUM_JOBS, is_train=True)

    training_samples = list(sorted(training_samples, key=lambda d: len(d["input_ids"])))
    valid_samples = list(sorted(valid_samples, key=lambda d: len(d["input_ids"])))

    train_dataset = FeedbackDataset(training_samples, args, tokenizer)
    valid_dataset = FeedbackDataset(valid_samples, args, tokenizer)

    num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)

    collate_fn = Collate(tokenizer, args)

    model = FeedbackModel(
        model_name=args.model,
        num_train_steps=num_train_steps,
        learning_rate=args.lr,
        num_labels=3,
        steps_per_epoch=len(train_dataset) / args.batch_size,
    )

    model = Tez(model)
    es = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join(args.output, f"model_f{args.fold}.bin"),
        patience=5,
        mode="min",
        delta=0.001,
        save_weights_only=True,
    )
    config = TezConfig(
        training_batch_size=args.batch_size,
        validation_batch_size=args.valid_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        fp16=True,
        step_scheduler_after="batch",
        val_strategy="batch",
        val_steps=1500,
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_collate_fn=collate_fn,
        valid_collate_fn=collate_fn,
        callbacks=[es],
        config=config,
    )


def predict(args):
    NUM_JOBS = 2
    seed_everything(42)
    df = pd.read_csv(os.path.join(args.input, "test.csv"))
    df.loc[:, "discourse_effectiveness"] = "Adequate"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = prepare_training_data(df, tokenizer, args, num_jobs=NUM_JOBS, is_train=False)
    samples = list(sorted(samples, key=lambda d: len(d["input_ids"])))

    dataset = FeedbackDataset(samples, args, tokenizer)
    num_train_steps = int(len(dataset) / args.batch_size / args.accumulation_steps * args.epochs)

    model = FeedbackModel(
        model_name=args.model,
        num_train_steps=num_train_steps,
        learning_rate=args.lr,
        num_labels=3,
        steps_per_epoch=len(dataset) / args.batch_size,
    )

    model = Tez(model)
    config = TezConfig(
        test_batch_size=args.batch_size,
        fp16=True,
    )
    model.load(os.path.join(args.output, f"model_f{args.fold}.bin"), weights_only=True, config=config)

    collate_fn = Collate(tokenizer, args)
    iter_preds = model.predict(dataset, collate_fn=collate_fn)
    preds = []
    for temp_preds in iter_preds:
        preds.append(temp_preds)

    preds = np.vstack(preds)

    sample_submission = pd.read_csv(os.path.join(args.input, "sample_submission.csv"))
    sample_submission.loc[:, "discourse_id"] = [x["discourse_id"] for x in samples]
    sample_submission.loc[:, "Ineffective"] = preds[:, 0]
    sample_submission.loc[:, "Adequate"] = preds[:, 1]
    sample_submission.loc[:, "Effective"] = preds[:, 2]
    sample_submission.to_csv(f"preds_{args.fold}.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.predict:
        predict(args)
    else:
        main(args)
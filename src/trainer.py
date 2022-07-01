# Initialize
NUM_JOBS = 12
seed_everything(42)
os.makedirs(cfg.output, exist_ok=True)



# Create fold
df = pd.read_csv(os.path.join(cfg.input, "train.csv"))
gkf = GroupKFold(n_splits=cfg.fold_num)
for fold, ( _, val_) in enumerate(gkf.split(X=df, groups=df.essay_id)):
    df.loc[val_ , "kfold"] = int(fold)

df["kfold"] = df["kfold"].astype(int)
df.groupby('kfold')['discourse_effectiveness'].value_counts()


# DataSet Preparation
train_df = df[df["kfold"] != cfg.val_fold].reset_index(drop=True)
valid_df = df[df["kfold"] == cfg.val_fold].reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
training_samples = prepare_training_data(train_df, tokenizer, cfg, num_jobs=NUM_JOBS, is_train=True)
valid_samples = prepare_training_data(valid_df, tokenizer, cfg, num_jobs=NUM_JOBS, is_train=True)

training_samples = list(sorted(training_samples, key=lambda d: len(d["input_ids"])))
valid_samples = list(sorted(valid_samples, key=lambda d: len(d["input_ids"])))

train_dataset = FeedbackDataset(training_samples, cfg, tokenizer)
valid_dataset = FeedbackDataset(valid_samples, cfg, tokenizer)

num_train_steps = int(len(train_dataset) / cfg.batch_size / cfg.accumulation_steps * cfg.epochs)

collate_fn = Collate(tokenizer, cfg)


# Model Preparation
model = FeedbackModel(
    model_name=cfg.model,
    num_train_steps=num_train_steps,
    learning_rate=cfg.lr,
    num_labels=3,
    steps_per_epoch=len(train_dataset) / cfg.batch_size,
    gpu_optimize_config=cfg.gpu_optimize_config,
)
model = Tez(model)


# Training
es = EarlyStopping(
    monitor="valid_loss",
    model_path=os.path.join(cfg.output, f"model_f{cfg.val_fold}.bin"),
    patience=5,
    mode="min",
    delta=0.001,
    save_weights_only=True,
)

train_config = TezConfig(
    training_batch_size=cfg.batch_size,
    validation_batch_size=cfg.valid_batch_size,
    gradient_accumulation_steps=cfg.accumulation_steps,
    epochs=cfg.epochs,
    fp16=cfg.gpu_optimize_config.fp16,
    step_scheduler_after="batch",
    val_strategy="batch",
    val_steps=cfg.val_steps,
)

model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    train_collate_fn=collate_fn,
    valid_collate_fn=collate_fn,
    callbacks=[es],
    config=train_config,
)
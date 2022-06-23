from sklearn.metrics import log_loss
import torch.nn.functional as F
def score(preds): return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}

"""Now we can create our model and trainer. HuggingFace uses the `TrainingArguments` class to set up arguments. We'll use a cosine scheduler with warmup. We'll use fp16 since it's much faster on modern GPUs, and saves some memory. We evaluate using double-sized batches, since no gradients are stored so we can do twice as many rows at a time."""

lr,bs = 8e-5,16
wd,epochs = 0.01,1

def get_trainer(dds):
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=wd, report_to='none')
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=3)
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokz, compute_metrics=score)

"""Let's train!"""

trainer = get_trainer(dds)
trainer.train()

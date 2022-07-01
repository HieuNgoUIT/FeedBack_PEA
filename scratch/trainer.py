from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import log_loss
import torch.nn.functional as F
import torch

def score(preds): 
    return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}


def get_trainer(config, dds, tokenizer):
    args = TrainingArguments('outputs', learning_rate=config.lr, warmup_ratio=0.1, lr_scheduler_type='cosine', #fp16=True,
        evaluation_strategy="epoch", per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=config.batch_size*2,
        num_train_epochs=config.epochs, weight_decay=0.01, report_to='none')

    model = AutoModelForSequenceClassification.from_pretrained(config.model, num_labels=3)
    
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokenizer, compute_metrics=score)


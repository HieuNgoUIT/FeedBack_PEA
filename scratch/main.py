from trainer import get_trainer
from data import main as prepare_data, tokenizer
from config import cfg


data_dict = prepare_data()
trainer = get_trainer(cfg, data_dict, tokenizer)
trainer.train()
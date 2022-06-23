from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer



"""Quiet down some of the warnings produced by HuggingFace Transformers:"""

warnings.simplefilter('ignore')
logging.disable(logging.WARNING)


model_nm = '../input/debertav3small'

"""We now get the tokenizer for our model:"""

tokz = AutoTokenizer.from_pretrained(model_nm)

"""For our baseline, we will concatenate the discourse type and the discourse text and pass to our model. We need to separate the discourse type and the discourse text so that our model knows which is which. We will use the special separator token that the tokenizer has:"""

sep = tokz.sep_token

def tok_func(x): 
    return tokz(x["inputs"], truncation=True)
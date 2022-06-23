
import datasets
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


df = pd.read_csv(path/'train.csv')
df.head()


"""Now let's create the input text:"""

df['inputs'] = df.discourse_type + sep +df.discourse_text

"""HuggingFace expects that the target is in a column to be called `label`, and also that the targets are numerical. We will categorize it and create a new column:"""

new_label = {"discourse_effectiveness": {"Ineffective": 0, "Adequate": 1, "Effective": 2}}
df = df.replace(new_label)
df = df.rename(columns = {"discourse_effectiveness": "label"})

"""Now let's create our `Dataset` object:"""

ds = Dataset.from_pandas(df)

"""To tokenize the data, let's create a function, since that's what `Dataset.map` will need:"""



"""Let's see what one example looks like when tokenized:"""

tok_func(ds[0])

"""We can now tokenize the  the input. We'll use `Dataset.map` to speed it up, and remove the columns we no longer need:"""

inps = "discourse_text","discourse_type"
tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','discourse_id','essay_id'))

"""Let's see all the columns:"""

tok_ds[0].keys()

"""Next we need to split the dataset into a training set and a validation set. We will split based on essays:"""

essay_ids = df.essay_id.unique()
np.random.seed(42)
np.random.shuffle(essay_ids)
essay_ids[:5]

"""We'll do a random 80%-20% split:"""

val_prop = 0.2
val_sz = int(len(essay_ids)*val_prop)
val_essay_ids = essay_ids[:val_sz]

is_val = np.isin(df.essay_id, val_essay_ids)
idxs = np.arange(len(df))
val_idxs = idxs[ is_val]
trn_idxs = idxs[~is_val]
len(val_idxs),len(trn_idxs)

"""We can use the `select` method of the `Dataset` object to create our splits:"""

dds = DatasetDict({"train":tok_ds.select(trn_idxs),
             "test": tok_ds.select(val_idxs)})

"""Here I put all of this into a single function, along with some extra code to deal with the test set (no split necessary):"""

def get_dds(df, train=True):
    ds = Dataset.from_pandas(df)
    to_remove = ['discourse_text','discourse_type','inputs','discourse_id','essay_id']
    tok_ds = ds.map(tok_func, batched=True, remove_columns=to_remove)
    if train:
        return DatasetDict({"train":tok_ds.select(trn_idxs), "test": tok_ds.select(val_idxs)})
    else: 
        return tok_ds

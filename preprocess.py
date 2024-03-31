import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    block_size = 32
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', max_length=70, truncation=True)

train_df = open('/content/TRAIN_DATA.txt').read().split('.')
train_df = [i for i in train_df if i != '']
train_df = pd.DataFrame({'text': train_df})
train_df.to_csv('TRAIN_DATA.csv', index=False, encoding='utf-8')

tokenized_datasets = load_dataset('csv', data_files='/content/TRAIN_DATA.csv')
tokenized_datasets.column_names
tokenized_datasets = tokenized_datasets["train"]

tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=8,
    num_proc=4,
)
tokenized_datasets.set_format("torch")

train_dataset, val_dataset = train_test_split(tokenized_datasets, shuffle=False, test_size=0.2, random_state=42)

train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)

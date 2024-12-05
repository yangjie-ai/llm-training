import os

import pandas as pd
import requests
import tiktoken
import numpy as np
from transformers import AutoTokenizer
import csv

def prepare_datacsv():
    df_q = pd.read_csv('question.csv')
    df_q = df_q.rename(columns={'content':'question'})
    df_a = pd.read_csv('answer.csv')
    df_a = df_a[['question_id','content']]
    df_a = df_a.rename(columns={'content': 'answer'})
    df_aq = pd.merge(df_q,df_a,how='left',on='question_id')
    # Limiting to the first 5000 rows
    # df_aq = df_aq.head(50)
    datalist = df_aq[['question', 'answer']].values
    txtdata = ""
    for data in datalist:
        txtdata += '\n'.join(data) + '\n'
    print(len(df_aq), len(txtdata))
    return txtdata
def prepare_data():
    data = prepare_datacsv()
    with open('input.txt', 'wt', encoding='utf-8') as f:
        f.write(data)
if __name__=='__main__':
    prepare_data()
    input_file_path = 'input.txt'
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

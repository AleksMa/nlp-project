import re

import nltk

from config import *
import pandas as pd
import json
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
label_set = set()

df = pd.read_json("data.txt", lines=True)
for i, r in df.iterrows():
    for j, val in enumerate(r['Targets']):
        tmp = val.split('.')
        if len(tmp) > 1:
            r['Targets'][j] = tmp[0]
        label_set.add(tmp[0])
    text = str()
    for _, val in enumerate(r['Data']):
        text += ' ' + val
    text = re.sub('https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', text)
    text = re.sub('\d+', '', text)
    text = re.sub('\-', '', text)
    text = re.sub(':', '', text)
    text = re.sub(',', '', text)
    text = re.sub('\"', '', text)
    text = re.sub('«', '', text)
    text = re.sub('%', '', text)
    text = re.sub('\w+\.ru', '', text)
    r['Data'] = text
    print(text)

df['id'] = 0
for it, val in enumerate(label_set):
    df[val] = 0.0

for i, r in df.iterrows():
    for j, val in enumerate(r['Targets']):
        df[val][i] = 1.0
    df['id'][i] = i

df.drop_duplicates(subset=['Data'], inplace=True)
"""
df = df[df.columns.drop(list(df.filter(regex='^((8|\+7)[\- ]?)?(\(?\d{3}\)?[\- ]?)?[\d\- ]{7,10}$')))]
df = df[df.columns.drop(list(df.filter(regex='^(http|https)://')))]
df = df[df.columns.drop(list(df.filter(regex='\d+')))]
df = df[df.columns.drop(list(df.filter(regex='\-')))]
df = df[df.columns.drop(list(df.filter(regex='\,')))]
df = df[df.columns.drop(list(df.filter(regex='\w+\.ru')))]
"""
c_text = []
for i in df['Data']:
    j = ' '.join([w for w in [w for w in i.split() if not w in stop_words]])
    c_text.append(j)

df['Data'] = c_text
df.describe()
df.head()

label = df.columns.tolist()[3:]
df[label].sum().sort_values().plot(kind="barh")
max_dfs = []
label_idx = 1
while label_idx < int(len(label) / DF_IDX_COUNT) * DF_IDX_COUNT:

    cur_df_idx = 0
    cur_df = pd.DataFrame()
    while cur_df_idx < DF_IDX_COUNT:
        append_df = df[df[label[label_idx + cur_df_idx]] == 1.0]
        if len(append_df) > 100:
            max_dfs += [(append_df, label[label_idx + cur_df_idx])]
        #max_dfs += [(append_df, label[label_idx + cur_df_idx])]
        cur_df = cur_df.append(append_df)
        cur_df_idx += 1

    cur_df = cur_df.drop('Targets', axis=1)
    rm_idx = 0
    while rm_idx < len(label):
        if rm_idx < label_idx or rm_idx >= label_idx + DF_IDX_COUNT:
            cur_df = cur_df.drop(label[rm_idx], axis=1)
        rm_idx += 1

    new_idx = ['Data', 'id'] + label[label_idx:label_idx + DF_IDX_COUNT]
    cur_df = cur_df.reset_index()
    cur_df = cur_df.drop('index', axis=1)
    cur_df_i = 0
    for cur_df_i, _ in cur_df.iterrows():
        cur_df['id'][cur_df_i] = cur_df_i

    cur_df = cur_df.reindex(new_idx, axis=1)
    cur_df.describe()
    cur_df.head()
    res = cur_df.to_json(orient="records")
    json_data = json.loads(res)
    with open('data/curdf' + str(label_idx) + '.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f)
    label_idx += DF_IDX_COUNT

cur_df = pd.DataFrame()
labels_max = []
for d, l in max_dfs:
    cur_df = cur_df.append(d)
    labels_max += [l]
print(labels_max)
for rm in label:
    if rm not in labels_max:
        cur_df.drop(rm, axis=1)

cur_df = cur_df.drop('Targets', axis=1)
new_idx = ['Data', 'id'] + labels_max
cur_df = cur_df.reset_index()
cur_df = cur_df.drop('index', axis=1)

cur_df_i = 0
for cur_df_i, _ in cur_df.iterrows():
    cur_df['id'][cur_df_i] = cur_df_i

cur_df = cur_df.reindex(new_idx, axis=1)
cur_df.describe()
cur_df.head()
res = cur_df.to_json(orient="records")
json_data = json.loads(res)
with open('df_top.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f)

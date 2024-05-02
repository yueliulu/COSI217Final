import json
import pandas as pd
import os
import random

directory = 'processed_data/json_file'

def load_json(directory):
    texts_train = []
    emojis_train = []
    texts_eval = []
    emojis_eval = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data = list(data.values())
            if len(data) > 1000:
                train_data = random.sample(data, 1000)
                eval_data = random.sample(data, 100)
            for item in train_data:
                texts_train.append(item['Plain text'])
                emojis_train.append(set(item['Emojis']))
            for item in eval_data:
                texts_eval.append(item['Plain text'])
                emojis_eval.append(set(item['Emojis']))

    emoji_set_train = set()
    for emoji in emojis_train:
        emoji_set_train.update(emoji)
    emoji_set_eval = set()
    for emoji in emojis_eval:
        emoji_set_eval.update(emoji)

    df_train = pd.DataFrame(0, index=pd.RangeIndex(len(texts_train)), columns=list(emoji_set_train))
    df_train['text'] = texts_train  # Add texts as a column in the DataFrame

    # Populate the DataFrame
    for i, emoji in enumerate(emojis_train):
        for emo in emoji:
            df_train.loc[i, emo] = 1

    df_eval = pd.DataFrame(0, index=pd.RangeIndex(len(texts_eval)), columns=list(emoji_set_eval))
    df_eval['text'] = texts_eval  # Add texts as a column in the DataFrame

    # Populate the DataFrame
    for i, emoji in enumerate(emojis_eval):
        for emo in emoji:
            df_eval.loc[i, emo] = 1

    return df_train, df_eval

df_train, df_eval = load_json(directory)
df_train.to_csv('train.csv', index=False)
df_eval.to_csv('eval.csv', index=False)
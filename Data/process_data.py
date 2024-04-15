import json
import pandas as pd

file_path1 = './processed_data/json_file/backhand_index_pointing_right.json'

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    emojis = []

    for key, value in data.items():
        texts.append(value['raw text'])
        emojis.append(set(value['emojis']))

    emoji_set = set()
    for emoji in emojis:
        emoji_set.update(emoji)
    print(len(emoji_set))

    df = pd.DataFrame(0, index=pd.RangeIndex(len(texts)), columns=list(emoji_set))
    df['text'] = texts  # Add texts as a column in the DataFrame

    # Populate the DataFrame
    for i, emoji in enumerate(emojis):
        for emo in emoji:
            df.loc[i, emo] = 1

    return df

df = load_json(file_path1)
df.to_csv('output.csv', index=False)
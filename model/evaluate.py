from utils import *
import numpy as np
import pandas as pd
from load_data import CustomDataset
from transformers import BertTokenizer
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import json

eval_path = "../Data/eval.csv"
emoji_classification_path = "../emoji_classification.json"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_true_labels(eval_path: str):
    df = pd.read_csv(eval_path)
    y_true = df.loc[:, emojis].values
    texts = df.loc[:, 'text'].values.astype(str)
    y_true_emotion = df.loc[:, ['Positive Emotion', 'Negative Emotion', 'Others']].values
    return y_true, texts, y_true_emotion

def predict_all(eval_path, threshold: float):
    df = pd.read_csv(eval_path)

    eval_dataset = CustomDataset(df, tokenizer, MAX_LEN, emojis)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    model = load_model('best_model.pt')
    model.eval()
    scores = []

    with torch.no_grad():
        for batch_idx, data in enumerate(eval_dataloader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            # targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            scores.append(outputs)
    scores = np.concatenate(scores, axis=0)
    y_pred = []
    for score in scores:
        pred = [1 if i >= threshold else 0 for i in score]
        if pred == [0] * 43:
            pred[int(np.argmax(score))] = 1
        y_pred.append(pred)
    y_pred = np.array(y_pred)
    return y_pred

def get_y_pred_emotion(y_pred, emoji_classification_path):
    with open(emoji_classification_path, 'r') as f:
        emoji_classes = json.load(f)
    y_pred_emotion = np.zeros((y_pred.shape[0], 3))

    # Map the categories to their indices based on the emojis list
    category_indices = {
        'Positive Emotion': [emojis.index(emoji) for emoji in emoji_classes['Positive Emotion'] if emoji in emojis],
        'Negative Emotion': [emojis.index(emoji) for emoji in emoji_classes['Negative Emotion'] if emoji in emojis],
        'Others': [emojis.index(emoji) for emoji in emoji_classes['Others'] if emoji in emojis]
    }

    # For each category, check if any emoji is 1 and assign to the respective column
    for index, category in enumerate(['Positive Emotion', 'Negative Emotion', 'Others']):
        # Use np.any with axis=1 to check across rows if any of the relevant indices is 1
        y_pred_emotion[:, index] = np.any(y_pred[:, category_indices[category]], axis=1).astype(int)
    return y_pred_emotion

def evaluate(y_true, y_pred, labels):
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels)
    return confusion_matrix, report

def calculate_acc(y_true_emotion, y_pred_emotion):
    equal_elements = y_true_emotion == y_pred_emotion

    # Calculate the sum of true values in each row and divide by the number of columns to get the score per row
    scores_per_row = np.sum(equal_elements, axis=1) / y_true_emotion.shape[1]

    # Calculate the average score across all rows
    average_score = np.mean(scores_per_row)
    return average_score


if __name__ == '__main__':
    y_true, texts, y_true_emotion = get_true_labels(eval_path)
    y_pred = predict_all(eval_path, threshold=0.1)
    y_pred_emotion = get_y_pred_emotion(y_pred, emoji_classification_path)
    acc = calculate_acc(y_true_emotion, y_pred_emotion)

    # # print first 10 examples
    # for i in range(10):
    #     gold_labels = y_true[i]
    #     pred_labels = y_pred[i]
    #     text = texts[i]
    #     gold = [emojis[j] for j in range(len(gold_labels)) if gold_labels[j] == 1]
    #     predict = [emojis[j] for j in range(len(pred_labels)) if pred_labels[j] == 1]
    #     print(text, gold, predict)

    confusion_matrix, report = evaluate(y_true, y_pred, labels=emojis)
    emotion_confusion_matrix, emotion_report = evaluate(y_true_emotion, y_pred_emotion, labels=['Positive Emotion',
                                                                                                'Negative Emotion',
                                                                                                'Others'])
    print("confusion matrix:\n", confusion_matrix)
    print("classification report:\n", report)
    print("acc:", acc)
    print("emotion classification report:\n", emotion_report)


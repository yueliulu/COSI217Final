from utils import *
import numpy as np
import pandas as pd
from load_data import CustomDataset
from transformers import BertTokenizer
from sklearn.metrics import multilabel_confusion_matrix, classification_report

eval_path = "../Data/eval.csv"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_true_labels(eval_path: str):
    df = pd.read_csv(eval_path)
    y_true = df.iloc[:, :-1].values
    texts = df.iloc[:, -1].values.astype(str)
    return y_true, texts

def predict_all(eval_path, threshold: float):
    df = pd.read_csv(eval_path)
    columns = df.columns[:-1]
    eval_dataset = CustomDataset(df, tokenizer, MAX_LEN, columns)
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


def evaluate(y_true, y_pred):
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=emojis)
    return confusion_matrix, report

if __name__ == '__main__':
    y_true, texts = get_true_labels(eval_path)
    y_pred = predict_all(eval_path, threshold=0.1)

    # # print first 10 examples
    # for i in range(10):
    #     gold_labels = y_true[i]
    #     pred_labels = y_pred[i]
    #     text = texts[i]
    #     gold = [emojis[j] for j in range(len(gold_labels)) if gold_labels[j] == 1]
    #     predict = [emojis[j] for j in range(len(pred_labels)) if pred_labels[j] == 1]
    #     print(text, gold, predict)

    confusion_matrix, report = evaluate(y_true, y_pred)
    print("confusion matrix:\n", confusion_matrix)
    print("classification report:\n", report)


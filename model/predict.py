from transformers import BertTokenizer
import numpy as np
from utils import *
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict(input: str, model, threshold):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encodings = tokenizer.encode_plus(
        input,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    model.eval()
    with torch.no_grad():
        input_ids = encodings['input_ids'].to(device, dtype=torch.long)
        attention_mask = encodings['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = encodings['token_type_ids'].to(device, dtype=torch.long)
        output = model(input_ids, attention_mask, token_type_ids)
        final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
        # print(final_output)
        # print(train_df.columns[1:].to_list()[int(np.argmax(final_output, axis=1))])
        potential_emoji = [emojis[i] for i in range(len(final_output[0])) if final_output[0][i] > threshold]
        if not potential_emoji:
            potential_emoji = [emojis[int(np.argmax(final_output, axis=1))]]
        return potential_emoji

def predict_with_trained_model(input):
    print("in predict")
    input = "Hello, how are you"
    model_dir = os.path.join(os.path.dirname(__file__))
    checkpoint_path = os.path.join(model_dir, 'best_model.pt')
    model = load_model(checkpoint_path)
    print("in predict")
    output = predict(input, model, threshold=0.1)
    return output

if __name__ == '__main__':
    input = "Hello, how are you"
    checkpoint_path = 'best_model.pt'
    model = load_model(checkpoint_path)
    output = predict(input, model, threshold=0.1)
    print(output)
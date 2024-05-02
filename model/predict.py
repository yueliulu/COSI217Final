import torch
from model import BERTClass
import torch.optim as optim
from transformers import BertTokenizer, BertModel

emojis = ['👉', '🥲', '😍', '🥳', '😤', '🥵', '😇', '🖕', '🙏', '☺️', '🍳', '🐣', '☀️', '🥚',
       '🤔', '🥰', '👻', '💀', '❤️', '😨', '🔥', '😂', '🐰', '✨', '😭', '😎', '😡', '🫠',
       '🥹', '😋', '✅', '🤍', '🐇', '🫡', '👀', '🎉', '✔️', '🤡', '👍', '😅', '💩', '😉',
       '🤣']
MAX_LEN = 128
NUM_CLASS = len(emojis)
DR = 0.2
HIDDEN_DIM = 768

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model(checkpoint_path):
    model = BERTClass(dr=DR, num_class=NUM_CLASS, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model, optimizer, start_epoch, valid_loss_min = load_ckp(checkpoint_path, model, optimizer)
    return model

def predict(input: str, model):

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
        potential_emoji = [emojis[i] for i in range(len(final_output[0])) if final_output[0][i] > 0.5]
        return potential_emoji


if __name__ == '__main__':
    input = "Hello, how are you"
    checkpoint_path = 'best_model.pt'
    model = load_model(checkpoint_path)
    output = predict(input, model)
    print(output)
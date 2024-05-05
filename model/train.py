import torch
import pandas as pd
import numpy as np
import sys
from transformers import BertTokenizer, BertModel
from model import BERTClass
from load_data import CustomDataset
from utils import save_ckp, emojis

train_path = "../Data/train.csv"
eval_path = "../Data/eval.csv"

df = pd.read_csv(train_path)


MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-05
NUM_CLASS = len(emojis)
DR = 0.2
HIDDEN_DIM = 768

train_size = 0.8
train_df = df.sample(frac=train_size, random_state=200).reset_index(drop=True)
val_df = df.drop(train_df.index).reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN, emojis)
valid_dataset = CustomDataset(val_df, tokenizer, MAX_LEN, emojis)

train_data_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_data_loader = torch.utils.data.DataLoader(valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BERTClass(dr=DR, num_class=NUM_CLASS, hidden_dim=HIDDEN_DIM)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
val_targets = []
val_outputs = []


def train_model(n_epochs, training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path):
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print('############# Epoch {}: Training Start   #############'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
            # print('yyy epoch', batch_idx)
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            # if batch_idx%5000==0:
            #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('before loss data in training', loss.item(), train_loss)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            # print('after loss data in training', loss.item(), train_loss)

        print('############# Epoch {}: Training End     #############'.format(epoch))

        print('############# Epoch {}: Validation Start   #############'.format(epoch))
        ######################
        # validate the model #
        ######################

        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            print('############# Epoch {}: Validation End     #############'.format(epoch))
            # calculate average losses
            # print('before cal avg train loss', train_loss)
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)
            # print training/validation statistics
            print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
            ))

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # save checkpoint
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            ## TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                valid_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                # torch.save(model.state_dict(), best_model_path)
                valid_loss_min = valid_loss

        print('############# Epoch {}  Done   #############\n'.format(epoch))

    return model

ckpt_path = "curr_ckpt.pt"
best_model_path = "best_model.pt"
trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, ckpt_path, best_model_path)

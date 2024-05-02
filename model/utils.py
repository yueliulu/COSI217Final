import torch
from model import BERTClass
import torch.optim as optim
import shutil

emojis = ['👉', '🥲', '😍', '🥳', '😤', '🥵', '😇', '🖕', '🙏', '☺️', '🍳', '🐣', '☀️', '🥚',
       '🤔', '🥰', '👻', '💀', '❤️', '😨', '🔥', '😂', '🐰', '✨', '😭', '😎', '😡', '🫠',
       '🥹', '😋', '✅', '🤍', '🐇', '🫡', '👀', '🎉', '✔️', '🤡', '👍', '😅', '💩', '😉',
       '🤣']
MAX_LEN = 128
NUM_CLASS = len(emojis)
DR = 0.2
HIDDEN_DIM = 768
BATCH_SIZE = 32

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, checkpoint['epoch'], valid_loss_min

def load_model(checkpoint_path):
    model = BERTClass(dr=DR, num_class=NUM_CLASS, hidden_dim=HIDDEN_DIM)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    model, start_epoch, valid_loss_min = load_ckp(checkpoint_path, model)
    return model
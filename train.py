import pickle
import argparse

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import NeuralStatistician
from data import OmniglotDataset
from eval import create_tests, evaluate

mps = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA: {device}')
elif torch.backends.mps.is_built():
    device = torch.device('mps')
    mps = True
    print('Using Metal')
else:
    device = torch.device('cpu')
    print('Using CPU')
    
with open('./data/chardata.pkl', 'rb') as f: objs = pickle.load(f)
val, val_labels = objs[2], objs[3]

def eval(model):
    eval_model = NeuralStatistician(batch_size=21, sample_size=1).to(device)
    eval_model.load_state_dict(model.state_dict())
    tests = create_tests(100, val, val_labels)
    _, correct = evaluate(tests, eval_model, 100)
    del eval_model
    torch.cuda.empty_cache()
    if mps: torch.mps.empty_cache()
    return correct

def plot(losses, val_losses, evals, prefix):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4)) 
    axs[0].plot(losses)
    axs[0].set_title('Loss')
    axs[1].plot(val_losses)
    axs[1].set_title('Val Loss')
    axs[2].plot(evals)
    axs[2].set_title('One-shot Classification')
    plt.savefig(f'figures/{prefix}-training.png')

def train(
    model, optim, loader, val_loader,
    epochs, checkpoint_at, eval_at, prefix,
    alpha
):
    
    model.train()
    train_losses = []
    val_losses = []
    evals = []
    weight = 1
    
    for epoch in range(epochs):
        print(f'Starting epoch: {epoch + 1}') 
        
        cum_loss = 0
        for batch in tqdm(loader):
            inputs = batch.to(device)
            loss = model.step(inputs, optim, (1 + weight) if alpha is not None else None)
            cum_loss += loss
            
        weight *= 0.5 # down-weight
       
        print(f'Loss: {cum_loss:.3e}')
        train_losses.append(cum_loss.cpu().detach().numpy())
        if torch.isnan(cum_loss):
            print('Encountered nan loss')
            break
        
        if (epoch + 1) % checkpoint_at == 0:
            torch.save(model.state_dict(), f'checkpoints/{prefix}-checkpoint-model-{epoch}')
            torch.save(optim.state_dict(), f'checkpoints/{prefix}-checkpoint-optim-{epoch}')
            
        if (epoch + 1) % eval_at == 0:
            model.eval()
           
            # evaluate val loss
            cum_val_loss = 0
            for batch in tqdm(val_loader):
                if batch.shape[0] != 16: continue
                inputs = batch.to(device)
                outputs = model(inputs)
                cum_val_loss += model.loss(*outputs)
            val_losses.append(cum_val_loss.cpu().detach().numpy())
            print(f'Val Loss: {cum_val_loss:.3e}')
            
            # evaluate one-shot classification 
            correct = eval(model)
            evals.append(correct)
            print(f'Correct: {correct:.3f}')
            
        model.train()
    
    plot(
        train_losses,
        val_losses,
        [e.cpu().detach().numpy() for e in evals],
        prefix
    )

# NOTE -- hyperparams are frozen for the most part
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--checkpoint-at', type=int, default=20)
    parser.add_argument('--eval-at', type=int, default=20)
    parser.add_argument('--from-checkpoint', type=int)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--alpha-weight', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset = OmniglotDataset('./data/chardata.pkl')
    loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    
    val_dataset = OmniglotDataset('./data/chardata.pkl', split_id=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)
    
    model = NeuralStatistician().to(device)
    optim = AdamW(model.parameters(), lr=args.lr)
    
    if args.from_checkpoint:
        model.load_state_dict(torch.load(f'checkpoints/checkpoint-model-{args.from_checkpoint}'))
        optim.load_state_dict(torch.load(f'checkpoints/checkpoint-optim-{args.from_checkpoint}'))
        
    train(
        model, optim, loader, val_loader,
        args.epochs, args.checkpoint_at, args.eval_at, args.prefix,
        args.alpha_weight
    )
    
if __name__ == '__main__':
    main()

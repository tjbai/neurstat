import argparse

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import NeuralStatistician
from data import OmniglotDataset


if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA: {device}')
# NOTE -- slow!
elif torch.backends.mps.is_built():
    device = torch.device('mps')
    print('Using Metal')
else:
    device = torch.device('cpu')
    print('Using CPU')

def plot(losses):
    plt.plot(losses)
    plt.title('Loss')
    plt.savefig('figures/loss.png')

def train(model, optim, loader, epochs, checkpoint_at):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        print(f'Starting epoch: {epoch + 1}') 
        
        cum_loss = 0
        for batch in tqdm(loader):
            inputs = batch.to(device)
            loss = model.step(inputs, optim)
            cum_loss += loss
       
        print(f'Loss: {(cum_loss):.3e}')
        losses.append(cum_loss)
        
        if (epoch + 1) % checkpoint_at == 0:
            torch.save(model.state_dict(), f'checkpoints/checkpoint-model-{epoch}')
            torch.save(optim.state_dict(), f'checkpoints/checkpoint-optim-{epoch}')
    
    plot(losses.cpu())

# NOTE -- hyperparams are frozen for the most part
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--checkpoint-at', type=int, default=20)
    parser.add_argument('--from-checkpoint', type=int)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset = OmniglotDataset('./data/chardata.pkl')
    loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    
    model = NeuralStatistician().to(device)
    optim = AdamW(model.parameters(), lr=args.lr)
    
    if args.from_checkpoint:
        model.load_state_dict(torch.load(f'checkpoints/checkpoint-model-{args.from_checkpoint}'))
        optim.load_state_dict(torch.load(f'checkpoints/checkpoint-optim-{args.from_checkpoint}'))
        
    train(model, optim, loader, args.epochs, args.checkpoint_at)
    
if __name__ == '__main__':
    main()
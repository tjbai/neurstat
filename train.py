import pickle
import argparse

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import NeuralStatistician
from data import ClassDataset
from evaluate import create_tests, evaluate

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
    eval_20_shot = NeuralStatistician(batch_size=1, sample_size=1).to(device)
    eval_20_cands = NeuralStatistician(batch_size=20, sample_size=1).to(device)
    eval_20_shot.load_state_dict(model.state_dict())
    eval_20_cands.load_state_dict(model.state_dict())
    
    eval_5_shot = NeuralStatistician(batch_size=1, sample_size=1).to(device)
    eval_5_cands = NeuralStatistician(batch_size=5, sample_size=1).to(device)
    eval_5_shot.load_state_dict(model.state_dict())
    eval_5_cands.load_state_dict(model.state_dict())
    
    tests_20 = create_tests(500, 1, 20, val, val_labels)
    tests_5 = create_tests(500, 1, 5, val, val_labels)
    
    _, correct_20 = evaluate(tests_20, eval_20_shot, eval_20_cands, 500)
    _, correct_5 = evaluate(tests_5, eval_5_shot, eval_5_cands, 500)
    
    del eval_20
    del eval_5
    torch.cuda.empty_cache()
    if mps: torch.mps.empty_cache()
    
    return correct_20, correct_5

def plot(losses, val_losses, evals, prefix):
    _, axs = plt.subplots(2, 2, figsize=(12, 8)) 
    
    axs[0][0].plot(losses)
    axs[0][0].set_title('Loss')
    axs[0][1].plot(val_losses)
    axs[0][1].set_title('Val Likelihood')
    
    axs[1][0].plot([eval[0] for eval in evals])
    axs[1][0].set_title('20-way classification')
    axs[1][1].plot([eval[1] for eval in evals])
    axs[1][1].set_title('5-way classification')
   
    plt.tight_layout() 
    plt.savefig(f'figures/{prefix}-training.png')

def train(
    model, optim, loader, val_loader,
    epochs, checkpoint_at, eval_at, prefix,
    alpha, split_classes
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
        if torch.isnan(cum_loss): break
        
        if (epoch + 1) % checkpoint_at == 0:
            torch.save(model.state_dict(), f'checkpoints/{prefix}-checkpoint-model-{epoch}')
            torch.save(optim.state_dict(), f'checkpoints/{prefix}-checkpoint-optim-{epoch}')
            
        if (epoch + 1) % eval_at == 0:
            model.eval()
           
            # evaluate val likelihood
            cum_val_loss = 0
            for batch in tqdm(val_loader):
                if batch.shape[0] != 16: continue
                inputs = batch.to(device)
                outputs = model(inputs)
                _, R_D = model.loss(*outputs)
                cum_val_loss += R_D
                
            # val_losses.append(cum_val_loss.cpu().detach().numpy())
            val_losses.append(cum_val_loss)
            print(f'Val Loss: {cum_val_loss:.3e}')
            
            # evaluate one-shot classification 
            if not split_classes:
                correct_20, correct_5 = eval(model)
                evals.append((correct_20.cpu().detach().numpy(), correct_5.cpu().detach().numpy()))
                print(f'Correct 20-way: {correct_20:.3f}')
                print(f'Correct 5-way: {correct_5:.3f}')
                
        model.train()
    
    plot(train_losses, val_losses, evals, prefix)

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
    parser.add_argument('--truncate', type=int)
    parser.add_argument('--split-classes', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    
    batch_size = (16, 120)[args.split_classes]
    sample_size = (5, 1)[args.split_classes]
    
    dataset = ClassDataset('./data/chardata.pkl', split_id=0, truncate=args.truncate, split_classes=args.split_classes)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ClassDataset('./data/chardata.pkl', split_id=1, split_classes=args.split_classes)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    
    model = NeuralStatistician(batch_size=batch_size, sample_size=sample_size).to(device)
    optim = AdamW(model.parameters(), lr=args.lr)
    
    if args.from_checkpoint:
        model.load_state_dict(torch.load(f'checkpoints/checkpoint-model-{args.from_checkpoint}'))
        optim.load_state_dict(torch.load(f'checkpoints/checkpoint-optim-{args.from_checkpoint}'))
        
    train(
        model, optim, loader, val_loader,
        args.epochs, args.checkpoint_at, args.eval_at, args.prefix,
        args.alpha_weight, args.split_classes
    )
    
if __name__ == '__main__':
    main()

import torch
import pickle
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models import NeuralStatistician

mps = False
if torch.cuda.is_available(): device = torch.device('cuda')
elif torch.backends.mps.is_built():
    device = torch.device('mps')
    mps = True
else: device = torch.device('cpu')

random.seed(42)

def kl(mu_q, logvar_q, mu_p, logvar_p):
    mu_p = mu_p.expand_as(mu_q) # minimize test TO train KLD
    logvar_p = logvar_p.expand_as(logvar_q)
    rat = ((mu_q - mu_p)**2 + torch.exp(logvar_q)) / torch.exp(logvar_p)
    return 0.5 * torch.sum(rat + logvar_p - logvar_q - 1, dim=1)

def create_tests(n, m, k, examples, labels):
    labels = labels - np.min(labels)
    uniq = list(set(labels))
    
    N = len(labels)
    C = len(uniq)
    indices = np.arange(len(labels)) 
    one_hot = np.zeros((N, C))
    for i, label in enumerate(labels): one_hot[i, label] = 1
    
    for _ in range(n):
        base_class, *noise = random.sample(uniq, k=k)
        res = random.sample(list(indices[one_hot[:, base_class].astype(bool)]), k=m+1)
        
        for i in noise:
            [d] = random.sample(list(indices[one_hot[:, i].astype(bool)]), k=1)
            res.append(d)
        
        inputs = torch.Tensor(examples[res]).view(m+k, 1, 28, 28).to(device)
        shots = inputs[:m].transpose(0, 1)
        candidates = inputs[m:]
            
        yield shots, candidates
        
        del shots
        del candidates
        
        torch.cuda.empty_cache()
        if mps: torch.mps.empty_cache()
    
def evaluate(tests, shot_model, cand_model, n):
    correct = 0
    shot_model.eval()
    cand_model.eval()
    
    for shots, candidates in tqdm(tests, total=n):
        mu_c, logvar_c, *_ = shot_model(shots)
        cand_mu_c, cand_logvar_c, *_ = cand_model(candidates)
        
        scores = kl(cand_mu_c, cand_logvar_c, mu_c, logvar_c)
        correct += torch.argmin(scores) == 0 
        
        del scores
        torch.cuda.empty_cache()
        if mps: torch.mps.empty_cache()
        
    return correct / n

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, help='total evaluation examples')
    parser.add_argument('--m', type=int, help='m-shot')
    parser.add_argument('--k', type=int, help='k-way')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', default=2) # test vs. val split
    return parser.parse_args()

def main():
    args = parse_args()
    
    shot_model = NeuralStatistician(batch_size=1, sample_size=args.m).to(device)
    shot_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    cand_model = NeuralStatistician(batch_size=args.k, sample_size=1).to(device)
    cand_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    with open('./data/chardata.pkl', 'rb') as f: objs = pickle.load(f)
    examples, labels = objs[2*args.split], objs[2*args.split+1]
    test = create_tests(args.n, args.m, args.k, examples, labels)
   
    error = evaluate(test, shot_model, cand_model, args.n)
    print(error.item())
    
if __name__ == '__main__':
    main()

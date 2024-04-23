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
    # mu_q = mu_q.expand_as(mu_p)
    # logvar_q = logvar_q.expand_as(logvar_p)
    mu_p = mu_p.expand_as(mu_q)
    logvar_p = logvar_p.expand_as(logvar_q)
    rat = ((mu_q - mu_p)**2 + torch.exp(logvar_q)) / torch.exp(logvar_p)
    return 0.5 * torch.sum(rat + logvar_p - logvar_q - 1, dim=1)

def is_correct(mu_c, logvar_c):
    mu_one_shot, mu_candidates = mu_c[0], mu_c[1:]
    logvar_one_shot, logvar_candidates = logvar_c[0], logvar_c[1:]
    scores = kl(mu_candidates, logvar_candidates, mu_one_shot, logvar_one_shot)
    cor = torch.argmin(scores) == 0
    del scores
    torch.cuda.empty_cache()
    if mps: torch.mps.empty_cache()
    return None, cor

def create_tests(n, examples, labels):
    labels = labels - np.min(labels)
    uniq = list(set(labels))
    
    N = len(labels)
    C = len(uniq)
    indices = np.arange(len(labels)) 
    one_hot = np.zeros((N, C))
    for i, label in enumerate(labels): one_hot[i, label] = 1
    
    for _ in range(n):
        base_class, *noise = random.sample(uniq, k=20)
        res = random.sample(list(indices[one_hot[:, base_class].astype(bool)]), k=2)
        
        for i in noise:
            [d] = random.sample(list(indices[one_hot[:, i].astype(bool)]), k=1)
            res.append(d)
            
        test = torch.Tensor(examples[res]).view(21, 1, 28, 28).to(device)
        yield test
        
        del test
        torch.cuda.empty_cache()
        if mps: torch.mps.empty_cache()
    
def evaluate(tests, model, n):
    model.eval()
    correct = 0
    all_scores = []
    for test in tqdm(tests, total=n):
        mu_c, logvar_c, *_ = model(test)
        scores, cor = is_correct(mu_c, logvar_c)
        all_scores.append(scores)
        correct += cor
    return all_scores, correct / n

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int)
    parser.add_argument('--from-checkpoint', type=Path)
    parser.add_argument('--split', default=2) # test vs. val split
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = NeuralStatistician(batch_size=21, sample_size=1).to(device) # 1 example + 20 candidates
    model.load_state_dict(torch.load(args.from_checkpoint, map_location=device))
    
    with open('./data/chardata.pkl', 'rb') as f: objs = pickle.load(f)
    examples, labels = objs[2*args.split], objs[2*args.split+1]
    test = create_tests(args.n, examples, labels)
   
    _, error = evaluate(test, model, args.n)
    print(error.item())
    
if __name__ == '__main__':
    main()

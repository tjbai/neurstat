import torch
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from models import NeuralStatistician, device
from data import OmniglotDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seed', type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed if args.seed else 42)
    
    model = NeuralStatistician(batch_size=5, sample_size=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
   
    dataset = OmniglotDataset('./data/chardata.pkl', split_id=1)
    example = dataset[np.random(len(dataset))]
    
    print(example.shape)
    
if __name__ == '__main__':
    main()
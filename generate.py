import torch
import argparse
import numpy as np
from pathlib import Path
from models import NeuralStatistician, device
from data import OmniglotDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--save-path', type=Path)
    return parser.parse_args()

# borrowed from https://github.com/conormdurkan/neural-statistician
def save_test_grid(inputs, samples, save_path):
    inputs = 1 - inputs.cpu().data.view(-1, 5, 1, 28, 28)
    reconstructions = samples.cpu().data.view(-1, 5, 1, 28, 28)
    images = torch.cat((inputs, reconstructions), dim=1).view(-1, 1, 28, 28)
    save_image(images, save_path, nrow=50)

def main():
    args = parse_args()
    
    model = NeuralStatistician(batch_size=5, sample_size=5).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
   
    dataset = OmniglotDataset('./data/chardata.pkl', split_id=1)
    loader = DataLoader(dataset, batch_size=5)
    example_batch = torch.Tensor(next(iter(loader))).to(device)
    
    samples = model.sample(example_batch)
    print(samples.shape)
    save_test_grid(example_batch, samples, args.save_path)
    
if __name__ == '__main__':
    main()
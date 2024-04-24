import pickle
import numpy as np
from torch.utils.data import Dataset

np.random.seed(42)

class OmniglotDataset(Dataset):
    
    def __init__(self, input_path, split_id=0):
        with open(input_path, 'rb') as f: objs = pickle.load(f)
        examples, labels = objs[2*split_id], objs[2*split_id+1]
        
        C = np.max(labels) + 1
        N = len(labels)
        one_hot = np.zeros((N, C)) # encode the class type for each example
        for i, label in enumerate(labels): one_hot[i, label] = 1
        
        idx = np.random.permutation(N)
        examples = examples[idx]
        labels = labels[idx]
        
        samples = []
        datasets = []
       
        # randomly sample 5 examples for each class 
        for i in range(C):
            mask = one_hot[:, i].astype(bool)
            class_examples = examples[mask]
            class_examples = class_examples[:5]
            if len(class_examples) < 5: continue
            samples.append(class_examples)
            datasets.append(one_hot[mask][:5])
        
        # NOTE -- might want to throw in some noising/masking here 
        inputs = np.concatenate(samples, axis=0)
        inputs = inputs.reshape(-1, 5, 1, 28, 28)
        targets = np.concatenate(datasets, axis=0)
        
        self.data = {'inputs': inputs, 'targets': targets}
        
    def __getitem__(self, i): return self.data['inputs'][i]
    
    def __len__(self): return len(self.data['inputs'])

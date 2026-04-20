import os
import random
import numpy as np
import torch


#Sets all random seeds to a fixed value for reproducibility, and returns a torch Generator with the same seed for use in dataloaders
def set_seed(seed = 1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return g

def seed_worker(worker_id, seed = 1234):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
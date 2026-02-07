from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
import torch
import torch.nn as nn




"""
DeepSAD loss implementation and helper functions. TODO: Turn this into hierarchical loss 
"""

#Takes in image-to-embedding model, returns size of embedding space
def get_output_size(model):
    return model._modules['vit']._modules['pooler']._modules['dense'].out_features


#Given the model and data, initialize the cluster centers c
def init_center_c(model, train_loader, device, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0

    model_size = get_output_size(model)
    c = torch.zeros(model_size, device=device)

    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            # get the inputs of the batch
            inputs = batch['pixel_values']
            inputs = inputs.to(device)
            outputs = model(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


#DeepSAD loss class. After initializing, it can be called as a function.
class DeepSADLoss():

    #Initialize using model, training data, and hyperparameters
    def __init__(self, model, train_data, device, eta, c = None, eps = 1e-6):
        if c is None:
            self.c = init_center_c(model, train_data, device)
        self.eta = eta
        self.eps = eps

    #Takes in embeddings and labels, returns a scalar value for the loss.
    def __call__(self, embeddings, labels):
        dist = torch.sum((embeddings - self.c) ** 2, dim=1)

        labels[labels == 0] = -1
        labels = -1 * labels

        losses = torch.where(labels == 5, dist, self.eta * ((dist + self.eps) ** labels.float()))
        loss = torch.mean(losses)
        return loss
    




"""
Triplet margin loss implementation
"""

#Takes in the margin hyperparameter, returns a function that takes embeddings and labels as inputs
def triplet_loss(margin = 0.2):
    def compute_triplet_loss(embeddings, labels):
        #Define the loss function and triplet miner
        loss_func = TripletMarginLoss(margin=margin)
        miner = TripletMarginMiner(margin=margin, type_of_triplets="semihard")

        #Use this to select the triplets that will be used in training
        mined_triplets = miner(embeddings, labels)

        # Pass embeddings, labels, and mined triplets
        loss = loss_func(embeddings, labels, mined_triplets)
        return loss

    return compute_triplet_loss




"""
TODO: Implement ArcFace loss
"""


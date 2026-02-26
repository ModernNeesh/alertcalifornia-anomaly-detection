from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
import torch
import torch.nn as nn

torch.manual_seed(1234)



#Takes in image-to-embedding model, returns size of embedding space
def get_output_size(model):
    return model._modules['vit']._modules['pooler']._modules['dense'].out_features


#Given the model and data, initialize the cluster centers c
def init_centers_c(model, train_loader, num_classes, device, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    
    
    n_samples = torch.zeros(num_classes, device = device)

    model_size = get_output_size(model)
    c = torch.zeros(num_classes, model_size, device=device)

    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            # get the inputs of the batch
            inputs = batch['pixel_values']
            labels = batch['labels']

            inputs = inputs.to(device)
            outputs = model(inputs)

            unique_labels, counts = torch.unique(labels, return_counts = True)
            for label in unique_labels:
                c[label] += torch.sum(outputs[labels == label], dim=0)

                count = counts[torch.argwhere(unique_labels == label)].item()

                n_samples[label] += count
            
    c = c.T / n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c.T

#Hierarchical SAD loss class. After initializing, it can be called as a function.
class HierarchicalSADLoss():

    #Initialize using model, training data, and hyperparameters
    def __init__(self, model, train_data, num_classes, device, eta, alpha, c = None, eps = 1e-6):
        self.num_classes = num_classes
        if c is None:
            self.c = init_centers_c(model, train_data, num_classes, device)
            self.c_norm = self.c[0, :]
        self.eta = eta
        self.eps = eps
        self.alpha = alpha
        self.device = device
        

    #Takes in embeddings and labels, returns a scalar value for the loss.
    def __call__(self, embeddings, labels):
        normal_dist = torch.sum((embeddings - self.c_norm) ** 2, dim=1)

        normal_classes = torch.tensor([0,1], device = self.device) #HARDCODED: Might be best to change later
        binary_labels = torch.where(torch.isin(labels, normal_classes), -1, 1)

        losses = (normal_dist + self.eps) ** binary_labels.float()
        loss = torch.mean(losses)

        anomalous_labels = torch.tensor([1,2,3], device=self.device) #HARDCODED: Might be best to change later
        anomalous_count = torch.isin(labels, anomalous_labels).shape[0]

        anomalous_dists = torch.empty(0, device=self.device)
        for label in anomalous_labels:
            anomalous_dists = torch.cat((anomalous_dists, torch.sum((embeddings[labels == label] - self.c[label]) ** 2, dim=1)))

        l_anom = torch.sum(anomalous_dists) / anomalous_count

        loss += self.alpha * l_anom
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


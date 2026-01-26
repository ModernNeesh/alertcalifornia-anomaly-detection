from transformers import ViTModel, ViTImageProcessor
from tqdm import tqdm
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
import torch
from sklearn.decomposition import PCA
import torch.nn as nn



#Train the model on the given data
def train_model(model, train_data, num_epochs, loss_func, optimizer, device, return_losses=True, save=True, name="params", path="weights/"):
    """
    model: The model to train
    train_data: The data to train on
    
    num_epochs: How many epochs to train for
    loss_func: The function with which to compute the loss
    optimizer: The optimizer to use during training
    return_losses: Whether to return the losses computed during training
    save: Whether to save the model
    name: Name of the model
    path: Where to save the model
    """
    losses = []
    for epoch in range(num_epochs):
        data_pbar = tqdm(enumerate(train_data))
        for i, batch in data_pbar:
            data_pbar.set_description(f"Processing batch {i} in epoch {epoch}")
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            embeddings = model(images)
            # Pass embeddings, labels, and mined triplets
            loss = loss_func(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
    if save == True:
        if name == "params":
            name = f"Epochs-{num_epochs}_Loss-{loss_func.__name__}_Optimizer-{type(optimizer).__name__}.pth"
        torch.save(model.state_dict(), path + name)
    
    if return_losses:
        return losses


#Returns triplet margin loss function with margin m
def triplet_loss(margin = 0.2):
    def compute_triplet_loss(embeddings, labels):
        loss_func = TripletMarginLoss(margin=margin)
        miner = TripletMarginMiner(margin=margin, type_of_triplets="semihard")

        mined_triplets = miner(embeddings, labels)

        # Pass embeddings, labels, and mined triplets
        loss = loss_func(embeddings, labels, mined_triplets)
        return loss

    return compute_triplet_loss


#Get embeddings of first batch of data loader
def get_batch_embeddings(model, data, device, return_ids=False):
    """
    model: Model to get embeddings with
    data: Dataloader to get embeddings from
    return_ids: Whether to return annotation ids of the batch
    """
    batch = next(iter(data))
    images = batch['pixel_values'].to(device)
    labels = batch['labels']

    embedding = model(images)

    if return_ids:
        return embedding, labels, batch['annotation_id']
    else:
        return embedding, labels


#Use PCA to reduce the dimensions of the given embeddings to the given number
def reduce_pca(embeddings, labels, dimensions = 2):  
    pca_model = PCA(n_components=dimensions)
    if type(embeddings) == torch.Tensor:
        reduced_embedding = pca_model.fit_transform(embeddings.to("cpu").detach().numpy())
    else: 
        reduced_embedding = pca_model.fit_transform(embeddings)
    if type(labels) == torch.Tensor:
        labels = labels.detach().numpy()

    return reduced_embedding, labels

#Class to make the encoder. It has the ViT architecture, just removes the classification head.
class ViTEmbeddingNet(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        
    def forward(self, pixel_values: torch.FloatTensor,labels: torch.LongTensor = None):
        outputs = self.vit(pixel_values)
        # Use [CLS] token (first token in the sequence) as embedding
        return outputs.last_hidden_state[:, 0]
    
#Classification head for model
class ClassificationHead(nn.Module):
    def __init__(self, input_dim = 768, num_classes = 5, hidden_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), # or nn.GELU(), etc.
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        x = self.norm(x)
        return self.head(x)
    
#Puts encoder and classification head together
class FullModel(nn.Module):
    def __init__(self, encoder, classification_head):
        self.encoder = encoder
        self.head = classification_head


    def forward(self, pixel_values: torch.FloatTensor,labels: torch.LongTensor = None):
        embeddings = self.encoder(pixel_values, labels)
        return self.head(embeddings)

def create_encoder(model_name = "google/vit-base-patch16-224"):
    model_name = "google/vit-base-patch16-224"
    vit = ViTModel.from_pretrained(model_name, dtype=torch.float32)
    return ViTEmbeddingNet(vit)

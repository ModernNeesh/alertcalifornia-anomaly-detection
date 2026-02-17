from tqdm import tqdm
import torch
from sklearn.decomposition import PCA

torch.manual_seed(1234)


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
    

def train_classification_head(model, train_data, num_epochs, criterion, optimizer, device, model_name="model", model_path = "weights/", save=True):

    losses = []

    head_criterion = criterion
    head_optimizer = optimizer

    head_name = model_path + model_name[:-4] + "_head.pth"



    num_epochs = 1
    for epoch in range(num_epochs):
        model.train() # Set model to training mode

        for batch in tqdm(train_data, desc = f"Processing batches in epoch {epoch}"):
            embeddings = batch['embeddings'].to(device).float()
            labels = batch['labels'].to(device).long()

            head_optimizer.zero_grad()
            outputs = model(embeddings)
            loss = head_criterion(outputs, labels)
            loss.backward()
            head_optimizer.step()

            losses.append(loss.item())
    if save == True:
        torch.save(model.state_dict(), head_name)

    return losses

def get_classification_accuracy(embeddings, labels, model, device):
    embeddings_tensor = torch.Tensor(embeddings.to_numpy()).to(device)
    labels_tensor = torch.Tensor(labels.to_numpy()).to(device)


    outputs = model(embeddings_tensor)

    return (torch.argmax(outputs, dim = -1) == labels_tensor).float().mean().item()




#Get embeddings of first batch of data loader
def get_batch_embeddings(model, data, device, return_ids=False):
    """
    model: Model to get embeddings with
    data: Dataloader to get embeddings from
    return_ids: Whether to return annotation ids of the batch
    """
    with torch.no_grad():
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




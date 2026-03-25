#Importing packages
import argparse
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from chromadb import PersistentClient as PersistentClient
from chromadb.errors import InternalError as CollectionError
import os

#library functions
import src.dataloading as dataloading
import src.data_vis as data_vis
import src.model_functions as model_functions
import src.loss_functions as loss_functions
import src.models as models
from src.seeds import set_seed

g = set_seed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running inference on the trained model.")
    parser.add_argument('--device', default="auto", choices=["auto", "cuda", "cpu"], 
                        help = "Which device to use")
    
    parser.add_argument('--objective', default="final", choices=["triplet", "deepsad", "hsad", "final"], 
                        help = "Which objective function to use")
    
    parser.add_argument("--image-dir", default = "camera_data/images/",
                        help="The directory where images to be processed are stored")
    
    parser.add_argument("--batch-size", default = 32, type=int,
                        help="The batch size to use for inference")
    
    parser.add_argument("--camera-data-dir", default="camera_data/",
                        help="The location the camera data is stored in")
    
    parser.add_argument("--data-csv-name", default="training_set_cameras_data.csv",
                        help="The location to store the camera data")


    parser.set_defaults()
    
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Using device:", device)



#Set directory to data and device for training
data_name = args.camera_data_dir + args.data_csv_name

#Set parameters for saving model
model_path = 'weights/'
model_name = "crossval_model.pth"

#Load data and create dataloaders
#TODO: Once cross validation splits are implemented, this will need to be updated to perform cross validation
if args.objective == "hsad":
    data = dataloading.get_data(data_name, args.image_dir, replace_images = False)

    labeled_data = data[data['choice'] > -1]

    full_train, _, _ = dataloading.get_train_val_test(data = data, output_csvs=False)

    train, val, test = dataloading.get_train_val_test(data = labeled_data, output_csvs=True)

    full_train_dataloader, _, _ = dataloading.get_train_val_test_dataloaders(full_train, val, test, generator = g)
    train_dataloader, val_dataloader, test_dataloader = dataloading.get_train_val_test_dataloaders(train, val, test, generator = g)
else:
    data = dataloading.get_data(data_name, args.image_dir, replace_images = False, binarize=True)
    train, val, test = dataloading.get_train_val_test(data = data, output_csvs=True)
    train_dataloader, val_dataloader, test_dataloader = dataloading.get_train_val_test_dataloaders(train, val, test, generator = g)

#Load encoder
encoder = models.create_encoder()
encoder.to(device)
encoder.load_state_dict(torch.load('weights/model_weights_camera_10-27-25.pth', map_location=device, weights_only=True));


#Set up loss function and optimizer based on objective
if args.objective == "triplet":
    loss_func = loss_functions.TripletMarginLoss(margin = 0.19)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-5) 
    num_epochs = 10
elif args.objective == "deepsad":
    loss_func = loss_functions.DeepSADLoss(model=encoder, train_data=train_dataloader, device = device, eta = 10)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-6, weight_decay=1e-7) 
    num_epochs = 1
elif args.objective == "hsad":
    loss_func = loss_functions.HierarchicalSADLoss(model=encoder, train_data=train_dataloader, num_classes = 4, device = device, eta = 0.01, alpha = 10)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-5, weight_decay=1e-6) 
    num_epochs = 5
elif args.objective == "final":
    loss_func = loss_functions.TripletMarginLoss(margin = 0.19)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-5) 
    num_epochs = 10


if args.objective in ["deepsad", "hsad"]:
    model_functions.train_model(encoder, train_data=full_train_dataloader, 
                                    num_epochs=num_epochs, loss_func=loss_func, 
                                    optimizer=optimizer, name = model_name, path = model_path, device=device, save=False, return_losses=False)
else:
    model_functions.train_model(encoder, train_data=train_dataloader, 
                                num_epochs=num_epochs, loss_func=loss_func, 
                                optimizer=optimizer, name = model_name, 
                                path = model_path, device=device, save=False, return_losses=False)

client = PersistentClient(path="embedding_data/") 

try:
    client.delete_collection(name="train_embeddings")
except Exception:
    pass

try:
    client.delete_collection(name="val_embeddings")
except Exception:
    pass


dataloading.save_full_embeddings(encoder, train_dataloader, 
                        "train_embeddings", persist_directory = "embedding_data/", 
                        device = device)


dataloading.save_full_embeddings(encoder, val_dataloader, 
                        "val_embeddings", persist_directory = "embedding_data/", 
                        device = device)
    
    
#Embeddings of training data, used to train the classification head
train_embeddings, train_labels, train_img_urls, train_a_ids = dataloading.load_full_embeddings(train, "train_embeddings", persist_directory = "embedding_data/")
train_embedding_dataloader = dataloading.embedding_to_dataloader(train_embeddings, train_labels, batch_size=32, generator = g)

#Embeddings of validation data
val_embeddings, val_labels, _, _ = dataloading.load_full_embeddings(val, "val_embeddings", persist_directory = "embedding_data/")



#Train the classification head on the embeddings of the training data, and evaluate on the validation data
classification_head = models.ClassificationHead()
classification_head.to(device)

head_criterion = nn.CrossEntropyLoss()
head_optimizer = optim.Adam(classification_head.parameters(), lr=1e-3) # Optimize only the new head

if args.objective in ["deepsad", "final"]:
    num_epochs_head = 10
else:
    num_epochs_head = 15

model_functions.train_classification_head(classification_head, train_embedding_dataloader, num_epochs=num_epochs_head, criterion=head_criterion, optimizer=head_optimizer, 
                           device=device, model_name=model_name, model_path = model_path, save=False)

#Report training and validation accuracy
classification_head.eval() 


#Function to calculate accuracy of classification head on given embeddings and labels
def get_accuracy(outputs, labels):
    return (torch.argmax(outputs, dim = -1) == labels).float().mean().item()


train_embeddings_tensor = torch.Tensor(train_embeddings.to_numpy()).to(device)
val_embeddings_tensor = torch.Tensor(val_embeddings.to_numpy()).to(device)

train_labels_tensor = torch.Tensor(train_labels.to_numpy()).to(device)
val_labels_tensor = torch.Tensor(val_labels.to_numpy()).to(device)


train_outputs = classification_head(train_embeddings_tensor)
val_outputs = classification_head(val_embeddings_tensor)

print(f"Training accuracy: {get_accuracy(train_outputs, train_labels_tensor)}")
print(f"Validation accuracy: {get_accuracy(val_outputs, val_labels_tensor)}")
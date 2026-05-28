#Importing packages
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from chromadb import PersistentClient as PersistentClient
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import time
import random

#library functions
import src.dataloading as dataloading
import src.model_functions as model_functions
import src.loss_functions as loss_functions
import src.models as models
from src.seeds import set_seed

start = time.perf_counter()

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
    
    parser.add_argument("--data-csv-name", default="coronado_hills_data.csv",
                        help="The location to store the camera data")


    parser.set_defaults()
    
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Using device:", device)

# Define the maximum value for a 32-bit signed integer
INT32_MAX = 2**32 - 1  

seed = random.randint(0, INT32_MAX)
g = set_seed(seed)


#Set directory to data and device for training
data_name = args.camera_data_dir + args.data_csv_name

#Set parameters for saving model
model_path = 'weights/'
model_name = "crossval_model.pth"


def get_labeled_data(df):
    """
    Helper function to get only the labeled data from the full dataset, which is used for training the classification head

    args:
        df (pd.DataFrame): the full dataset

    """
    return df[df['choice'] > -1]


def get_accs_of_fold(train, val, train_dataloader, val_dataloader, full_train_dataloader = None):
    #Load encoder
    encoder = models.create_encoder()
    encoder.to(device)


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
    
    def get_recall_precision(outputs, labels):
        predictions = torch.argmax(outputs, dim = -1).detach().cpu().numpy().copy()
        recall, precision, _, _ =  precision_recall_fscore_support(predictions, labels, average='weighted')
        return recall, precision




    train_embeddings_tensor = torch.Tensor(train_embeddings.to_numpy()).to(device)
    val_embeddings_tensor = torch.Tensor(val_embeddings.to_numpy()).to(device)

    train_labels_tensor = torch.Tensor(train_labels.to_numpy()).to(device)
    val_labels_tensor = torch.Tensor(val_labels.to_numpy()).to(device)


    train_outputs = classification_head(train_embeddings_tensor)
    val_outputs = classification_head(val_embeddings_tensor)

    train_acc = get_accuracy(train_outputs, train_labels_tensor)
    val_acc = get_accuracy(val_outputs, val_labels_tensor)
    recall, precision = get_recall_precision(val_outputs, val_labels)
    return train_acc, val_acc, recall, precision



#Load data and create dataloaders
stratification_cols = ['choice']
num_folds = 6

#If the objective is hierarchical SAD, data must be binarized
if args.objective == "hsad":
    binarize = False
    include_location = False
    inject_data = pd.read_csv("camera_data/coronado_labels_to_inject.csv")
elif args.objective == "final":
    binarize = True
    include_location = True
    inject_data = None
elif args.objective == "deepsad":
    binarize = True
    include_location = False
    inject_data = pd.read_csv("camera_data/coronado_labels_to_inject.csv")
else:
    binarize = True
    include_location = False
    inject_data = None

data = dataloading.get_data(data_name, args.image_dir, replace_images = False, binarize=binarize, include_location=include_location, inject_data = inject_data)

#For semi-supervised objectives, separately get full data folds and separate out labeled data. For supervised objectives, all data is labeled so only get labeled data folds
if args.objective in ["deepsad", "hsad"]:
    
    #Additionally stratify by whether the data is labeled or not, to ensure stratification remains after removing unlabeled data
    data['islabeled'] = data['choice'].apply(lambda x: 0 if x == -1 else 1)
    stratification_cols = ['choice', 'islabeled']

    full_folds = dataloading.get_k_folds(data, k=num_folds, strat_cols=stratification_cols)
    labeled_folds = [get_labeled_data(fold) for fold in full_folds]
else:
    #Stratify by labels for final model
    if args.objective == "final":
        stratification_cols = ['choice', 'location']
    #The other objectives are supervised, so all data is labeled
    full_folds = None
    labeled_folds = dataloading.get_k_folds(data, k=num_folds, strat_cols=stratification_cols)

train_accs = np.array([])
val_accs = np.array([])
recalls = np.array([])
precisions = np.array([])

for i in range(num_folds-1): #We hold out the final fold as a test set
    val_fold_idx = i

    #Get folds for this iteration of cross validation. All folds except the validation and test fold are used for training
    labeled_train_folds = [fold for j, fold in enumerate(labeled_folds[:-1]) if j != val_fold_idx]
    labeled_val_fold = labeled_folds[val_fold_idx]

    #For semi-supervised objectives, get the full training folds as well, which include the unlabeled data. For supervised objectives, full_folds is None so this step is skipped.
    if full_folds is not None:
        full_train_folds = [fold for j, fold in enumerate(full_folds[:-1]) if j != val_fold_idx]

    #Concatenate folds together into a single dataframe
    labeled_train_data = pd.concat(labeled_train_folds, ignore_index=True)
    val_data = labeled_val_fold
    full_train_data = pd.concat(full_train_folds, ignore_index=True) if full_folds is not None else None

    #Create dataloaders for this fold, and get training and validation accuracies. 
    train_dataloader = dataloading.pipe_to_dataloader(labeled_train_data, batch_size=32, generator = g)
    val_dataloader = dataloading.pipe_to_dataloader(labeled_val_fold, batch_size=32, generator = g)

    if full_folds is not None:
        full_train_dataloader = dataloading.pipe_to_dataloader(full_train_data, batch_size=32, generator = g)
        train_acc, val_acc, recall, precision = get_accs_of_fold(labeled_train_data, val_data, train_dataloader, val_dataloader, full_train_dataloader)
    else:
        train_acc, val_acc, recall, precision = get_accs_of_fold(labeled_train_data, val_data, train_dataloader, val_dataloader)

    #Update list of training and validation accuracies across folds
    train_accs = np.append(train_accs, train_acc)
    val_accs = np.append(val_accs, val_acc)
    recalls = np.append(recalls, recall)
    precisions = np.append(precisions, precision)

#Get metrics for test fold
test_fold_idx = num_folds - 1
labeled_train_folds = [fold for j, fold in enumerate(labeled_folds) if j != test_fold_idx]
labeled_test_fold = labeled_folds[test_fold_idx]

#For semi-supervised objectives, get the full training folds as well, which include the unlabeled data. For supervised objectives, full_folds is None so this step is skipped.
if full_folds is not None:
    full_train_folds = [fold for j, fold in enumerate(full_folds) if j != test_fold_idx]

#Concatenate folds together into a single dataframe
labeled_train_data = pd.concat(labeled_train_folds, ignore_index=True)
test_data = labeled_test_fold
full_train_data = pd.concat(full_train_folds, ignore_index=True) if full_folds is not None else None

#Create dataloaders for this fold, and get training and validation accuracies. 
train_dataloader = dataloading.pipe_to_dataloader(labeled_train_data, batch_size=32, generator = g)
test_dataloader = dataloading.pipe_to_dataloader(labeled_test_fold, batch_size=32, generator = g)

if full_folds is not None:
    full_train_dataloader = dataloading.pipe_to_dataloader(full_train_data, batch_size=32, generator = g)
    final_train_acc, test_acc, final_recall, final_precision = get_accs_of_fold(labeled_train_data, test_data, train_dataloader, test_dataloader, full_train_dataloader)
else:
    final_train_acc, test_acc, final_recall, final_precision = get_accs_of_fold(labeled_train_data, test_data, train_dataloader, test_dataloader)


with open(r"outputs/cross_validation_results.txt", "a") as f:
    f.write(f"{args.objective}, {train_accs.mean()}, {train_accs.std()}," +
            f"{val_accs.mean()}, {val_accs.std()}, {recalls.mean()}, " +
            f"{recalls.std()}, {precisions.mean()}, {precisions.std()}," +
            f"{final_train_acc}, {test_acc}, {final_recall}, {final_precision}, {seed}")
    f.write("\n")

end = time.perf_counter()
total_seconds = end - start

# Manually convert seconds to hours and minutes
if total_seconds >= 3600:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
else:
    minutes, seconds = divmod(total_seconds, 60)
    print(f"Runtime: {int(minutes)}m {seconds:.2f}s")


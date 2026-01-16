#Importing packages
import argparse
import os
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from chromadb import PersistentClient as PersistentClient
from chromadb.errors import InternalError as CollectionError

#library functions
import helper_code.dataloading as dataloading
import helper_code.data_vis as data_vis
import helper_code.model_functions as model_functions


torch.manual_seed(1234)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple calculator script.")
    parser.add_argument("--camera-data-dir", default="camera_data/",
                        help="The location to store the camera data")
    
    parser.add_argument("--labels-csv-name", default = "coronado_hills_data.csv",
                        help="The csv file with image paths and labels, imported from Label Studio")
    
    parser.add_argument("--image-dir", default = "camera_data/images/",
                        help="The directory to save images to")
    
    parser.add_argument('--download-imgs', dest = "image_download", action='store_true', help = "Download new image data")
    parser.add_argument('--keep-imgs', dest='image_download', action='store_false', help = "Use existing image data")

    parser.add_argument("--model-path", default = "weights/", 
                        help = "The directory to store the model to, or load it from")
    
    parser.add_argument("--model-name", default = "model_weights.pth", 
                        help = "Name of the model's weights")
    
    parser.add_argument('--train-model', dest = "model_train", action='store_true', help = "Train a new model")
    parser.add_argument('--load-model', dest='model_train', action='store_false', help = "Load an existing model's weights")
    parser.add_argument('--device', default="auto", choices=["auto", "cuda", "cpu"], help = "Which device to use")

    parser.add_argument("--collection-dir", default = "embedding_data/", 
                        help = "The directory to save embeddings to")
    

    parser.set_defaults(image_download=True, model_train=True, embedding_save = True)
    
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Using device:", device)


#Load the data
print(f"Loading data...")
labels_csv = args.camera_data_dir + args.labels_csv_name

data = dataloading.get_data(labels_csv, args.image_dir, replace_images = args.image_download)

train, val, test = dataloading.get_train_val_test(data = data, output_csvs=True)

train_dataset, val_dataset, test_dataset = dataloading.get_datasets(train, val, test)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, pin_memory=True)


print(f"Data loading complete.")
#Train the model
print(f"Training model {args.model_name}...")

encoder = model_functions.create_encoder()
encoder.to(device)


if args.model_train:
    num_epochs = 1
    loss_func = model_functions.triplet_loss(margin=0.19)
    optimizer = optim.Adam(encoder.parameters(), lr=2e-5) 

    model_functions.train_model(encoder, train_data=train_dataloader, 
                                num_epochs=num_epochs, loss_func=loss_func, 
                                optimizer=optimizer, name = args.model_name, path = args.model_path, device=device)
else:
    encoder.load_state_dict(torch.load(args.model_path + args.model_name, map_location=device, weights_only=True))
encoder.eval()


print(f"Model training complete; model is located at {args.model_path + args.model_name}.")
#Save embeddings
print("Saving embeddings...")



client = PersistentClient(path=args.collection_dir) 

try:
    client.delete_collection(name="train_embeddings")
except Exception:
    pass

try:
    client.delete_collection(name="val_embeddings")
except Exception:
    pass

try:
    client.delete_collection(name="test_embeddings")
except Exception:
    pass

dataloading.save_full_embeddings(encoder, train_dataloader, 
                        "train_embeddings", persist_directory = args.collection_dir, 
                        device = device)


dataloading.save_full_embeddings(encoder, val_dataloader, 
                        "val_embeddings", persist_directory = args.collection_dir, 
                        device = device)

dataloading.save_full_embeddings(encoder, test_dataloader, 
                        "test_embeddings", persist_directory = args.collection_dir, 
                        device = device)
    
    
#Embeddings of training data, used to train the classification head
train_embeddings, train_labels, _, _ = dataloading.load_full_embeddings(train, "train_embeddings", persist_directory = args.collection_dir)
train_embedding_dataset = dataloading.CustomEmbeddingDataset(train_embeddings, train_labels)
train_embedding_dataloader = DataLoader(train_embedding_dataset, batch_size=64, shuffle=True, pin_memory=True)

#Embeddings of validation data, used to display results
val_embeddings, val_labels, _, _ = dataloading.load_full_embeddings(val, "val_embeddings", persist_directory = args.collection_dir)
val_embedding_dataset = dataloading.CustomEmbeddingDataset(val_embeddings, val_labels)
val_embedding_dataloader = DataLoader(val_embedding_dataset, batch_size=64, shuffle=True, pin_memory=True)

#Embeddings of test data, used to evaluate classification head
test_embeddings, test_labels, _, _ = dataloading.load_full_embeddings(test, "test_embeddings", persist_directory = args.collection_dir)
test_embedding_dataset = dataloading.CustomEmbeddingDataset(test_embeddings, test_labels)
test_embedding_dataloader = DataLoader(test_embedding_dataset, batch_size=64, shuffle=True, pin_memory=True)

print(f"Embedding loading complete. Training and test data embeddings are saved at {args.collection_dir} under the \
    names 'train_embeddings' and 'test_embeddings' respectively.")
#Train the classification head
print("Training classification head...")


classification_head = model_functions.ClassificationHead()
classification_head.to(device)

head_criterion = nn.CrossEntropyLoss()
head_optimizer = optim.Adam(classification_head.parameters(), lr=1e-4) # Optimize only the new head

head_name = args.model_path + args.model_name[:-4] + "_head.pth"

if (not args.model_train) and os.path.exists(head_name + ".pth"):
    classification_head.load_state_dict(torch.load(head_name, weights_only=True))

else:
    num_epochs = 1
    for epoch in range(num_epochs):
        classification_head.train() # Set model to training mode

        for batch in tqdm(train_embedding_dataloader, desc = f"Processing batches in epoch {epoch}"):
            embeddings = batch['embeddings'].to(device).float()
            labels = batch['labels'].to(device).long()

            head_optimizer.zero_grad()
            outputs = classification_head(embeddings)
            loss = head_criterion(outputs, labels)
            loss.backward()
            head_optimizer.step()
    torch.save(classification_head.state_dict(), head_name)


#Report training and validation accuracy
classification_head.eval()

print("Classification head training complete.")
#Report training and validation accuracy


def get_accuracy(embeddings, labels, model):
    embeddings_tensor = torch.Tensor(embeddings.to_numpy()).to(device)

    outputs = model(embeddings_tensor)

    labels_tensor = torch.Tensor(labels).to(device)

    accuracy = (torch.argmax(outputs, dim = -1) == labels_tensor).float().mean().item()

    return accuracy


print(f"Training accuracy: {get_accuracy(train_embeddings, train_labels, classification_head)}")
print(f"Test accuracy: {get_accuracy(test_embeddings, test_labels, classification_head)}")


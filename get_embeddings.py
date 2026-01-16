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
    parser.add_argument("--embeddings-name", default = "embeddings", 
                        help = "The name of the embeddings")
    

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

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, pin_memory=True)


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


embeddings_name = args.embeddings_name

client = PersistentClient(path=args.collection_dir) 

try:
    client.delete_collection(name=embeddings_name + "_train")
except Exception:
    pass

try:
    client.delete_collection(name=embeddings_name + "_val")
except Exception:
    pass

try:
    client.delete_collection(name=embeddings_name + "_test")
except Exception:
    pass


embeddings_name = args.embeddings_name

dataloading.save_full_embeddings(encoder, train_dataloader, 
                        embeddings_name + "_train", persist_directory = args.collection_dir, 
                        device = device)


dataloading.save_full_embeddings(encoder, val_dataloader, 
                        embeddings_name + "_val", persist_directory = args.collection_dir, 
                        device = device)

dataloading.save_full_embeddings(encoder, test_dataloader, 
                        embeddings_name + "_test", persist_directory = args.collection_dir, 
                        device = device)
    
    
print(f"Embeddings saved at {args.collection_dir}!")
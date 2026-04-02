#Importing packages
import argparse
import os
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np

from chromadb import PersistentClient as PersistentClient
from chromadb.errors import InternalError as CollectionError

#library functions
import src.dataloading as dataloading
import src.data_vis as data_vis
import src.model_functions as model_functions
import src.loss_functions as loss_functions
import src.models as models
from src.seeds import set_seed, seed_worker

g = set_seed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running inference on the trained model.")
    parser.add_argument("--model-name", default="final_model.pth",
                        help="The name of the model to run inference on")
    
    parser.add_argument('--device', default="auto", choices=["auto", "cuda", "cpu"], 
                        help = "Which device to use")
    
    parser.add_argument("--model-path", default = "weights/", 
                        help = "The directory to store the model to, or load it from")
    
    parser.add_argument("--image-dir", default = "camera_data/images/",
                        help="The directory where images to be processed are stored")
    
    parser.add_argument("--batch-size", default = 32, type=int,
                        help="The batch size to use for inference")
    
    parser.add_argument("--camera-data-dir", default="camera_data/",
                        help="The location the camera data is stored in")
    
    parser.add_argument("--data-csv-name", default="training_set_cameras_data.csv",
                        help="The location to store the camera data")
    
    parser.add_argument("--test-throughput", default=True, type=bool,
                        help="Whether to test throughput of the model")
    
    parser.add_argument("--clear-previous-results", default = True, type= bool,
                        help = "Whether to clear previous results before running inference")
    
    parser.add_argument("--use-sanitized-data", default = True, type= bool,
                        help = "Whether to use sanitized camera data. LEAVE THIS AS TRUE UNLESS YOU HAVE THE UNSANITIZED CAMERA DATA")


    parser.set_defaults()
    
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Using device:", device)


model_dir = args.model_path + args.model_name
head_name = model_dir[:-4] + "_head.pth"

if args.use_sanitized_data:
    args.data_csv_name = "sanitized_camera_data.csv"
data_name = args.camera_data_dir + args.data_csv_name

num_classes = 2 

#Load the model's encoder and classification head weights, and put them together into the full model

encoder = models.create_encoder()
encoder.load_state_dict(torch.load(model_dir, map_location=device, weights_only=True))

classification_head = models.ClassificationHead(num_classes=num_classes)
classification_head.load_state_dict(torch.load(head_name, map_location=device, weights_only=True))

full_model = models.FullModel(encoder, classification_head)
full_model.to(device)




#Load the data to run inference on, and put it into a dataloader

start_time = time.perf_counter()

if args.use_sanitized_data:
    data = dataloading.get_data(data_name, args.image_dir, replace_images = args.test_throughput, binarize = False)
else:
    data = dataloading.get_data(data_name, args.image_dir, replace_images = args.test_throughput, binarize = True)


inference_dataloader = dataloading.get_inference_dataloader(data, batch_size=args.batch_size, generator=g)

end_time = time.perf_counter()

num_batches = len(inference_dataloader)
loading_time = end_time - start_time

print(f"Data loading and dataloader creation took {loading_time:.4f} seconds. \
      Average time per batch: {loading_time/num_batches:.4f} seconds.")

inference_time = 0


save_path = "outputs/inference_results.csv"
if os.path.exists(save_path):
    if args.clear_previous_results:
        os.remove(save_path)

#Write results to file in the form of: id, img_url, img_path, predicted_label, predicted_label_name
with open("outputs/inference_results.csv", "a") as f:
    for batch in tqdm(inference_dataloader, desc = "Running inference on batches"):
        start_time = time.perf_counter()
        images = batch['pixel_values'].to(device)
        outputs = full_model(images)
        end_time = time.perf_counter()

        inference_time += (end_time - start_time)

        ids = np.array(batch['id'])
        img_urls = np.array(batch['img_url'])
        img_paths = np.array(batch['img_path'])
        outputs_list = torch.argmax(outputs, dim = -1).cpu().detach().numpy()
        outputs_name = np.where(outputs_list == 0, "Normal", "Abnormal")

        stacked = np.hstack([ids.reshape(-1, 1), img_urls.reshape(-1, 1), img_paths.reshape(-1, 1), outputs_list.reshape(-1, 1), outputs_name.reshape(-1, 1)])

        np.savetxt(f, stacked, delimiter=',', fmt = "%s")
        f.write("\n")

print(f"Total inference time: {inference_time:.4f} seconds. Average time per batch: {inference_time/num_batches:.4f} seconds.")

if args.test_throughput:
    with open("outputs/throughput_results.csv", "a") as f:
        f.write(f"{args.batch_size}, {loading_time/num_batches:.4f}, {inference_time/num_batches:.4f}, {(inference_time+loading_time)/num_batches:.4f}, {(inference_time + loading_time):.4f}")
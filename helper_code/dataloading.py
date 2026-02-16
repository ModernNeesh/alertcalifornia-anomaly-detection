import pandas as pd
import requests
import os
from sklearn.model_selection import train_test_split
import time
import torch
import chromadb
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc
from requests.adapters import HTTPAdapter, Retry
import numpy as np
from helper_code.seeds import set_seed, seed_worker


#Helper function to get annotation result from differently formatted data
def get_annotation_result(x):
    if len(x) < 1:
        return np.nan
    else:
        result = x[0]['result']

        if len(result) < 1:
            return np.nan
        else:
            return result[0]['value']['choices'][0]



#Helper function to get annotation result from differently formatted data
def get_annotation_result(x):
    if len(x) < 1:
        return np.nan
    else:
        result = x[0]['result']

        if len(result) < 1:
            return np.nan
        else:
            return result[0]['value']['choices'][0]



#Get the label, image URL, and annotation id of the images
def get_data_urls(labels_csv, binarize = False):
    """
    labels_csv: The annotation file exported from LabelStudio, in csv form
    
    """
    if 'json' in labels_csv:
        raw_data = pd.read_json(labels_csv)



    data["choice"] = data["choice"].fillna('0').str.extract(r"(\d)").astype(int) - 1

    if binarize:
        data["choice"] = (data["choice"] > 1).astype(int)

    return data


#Download the images from their URLs
def download_images(data, image_dir, replace_images):
    """
    data: The dataset of image urls, usually the output of the get_data_urls function
    """

    # Set up a single persistent session (much faster + API friendly)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'DSC180 Capstone B07: ALERTCalifornia'
    })

    # Add robust retry logic (prevents failures on 503 / 504 / timeouts)
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))


    for i in tqdm(range(len(data))):
        url = data.iloc[i]['image']
        ann_id = data.iloc[i]['annotation_id']
        img_path = f"{image_dir}img_{ann_id}.jpg"

        # Skip if already downloaded
        if os.path.exists(img_path):
            if not replace_images:
                continue
            else:
                os.remove(img_path)

        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                with open(img_path, "wb") as f:
                    f.write(resp.content)
        except Exception as e:
            print(f"Error for ID {ann_id}: {e}")
            continue




#Get a dataframe of the image paths and their corresponding annotation ids
def get_images_df(image_dir):
    """
    img_dir: The directory of the images
    """
    img_paths = image_dir + pd.Series([path for path in os.listdir(image_dir) if path != ".gitkeep"])
    
    annotations = img_paths.str.extract(r"(\d+)")[0].astype(int).dropna()

    return pd.DataFrame({"annotation_id" : annotations, 
                         "img_path" : img_paths})


#Gather the data for image labels and paths into one big DataFrame
def get_data(labels_csv, image_dir, replace_images = False, binarize = False):
    """
    labels_csv: The annotation file exported from LabelStudio, in csv form

    image_dir: The directory for the images to be saved to/gathered from

    replace_images: Whether to replace the images currently in the directory
    """
    url_data = get_data_urls(labels_csv, binarize = binarize)

    download_images(url_data, image_dir, replace_images)

    image_df = get_images_df(image_dir)

    full_data = url_data.merge(image_df, left_on = "annotation_id", right_on="annotation_id")

    full_data = full_data.get(["choice", "img_path", "annotation_id", "image"])
    full_data['timestamp'] = full_data['image'].str.extract(r"https:\/\/tools\.alertcalifornia\.org\/fireframes5\/digitalpath-redis\/[^\/]+\/\d{4}\/\d{3}\/\d{2}\/(\d+)\.")

    return full_data

#Split a DataFrame into train, validation, and test splits, or pull those splits from existing CSV files
def get_train_val_test(data = None, df_dir = None, output_csvs = False, csv_output_dir = "camera_data/dataframes/"):
    """
    data: The DataFrame to split

    df_dir: The directory to the existing CSV files, if they exist

    output_csvs: Whether to output the train, validation, and test dataframes to a new file

    csv_output_dir: Where to output the dataframes
    """
    if df_dir is not None:
        train = pd.read_csv(df_dir + "train")
        val = pd.read_csv(df_dir + "val")
        test = pd.read_csv(df_dir + "test")
    else:
        if type(data) == type(None):
            raise ValueError("Must include dataframe to split")
        # X: features, y: target variable
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data[['img_path', 'image', 'annotation_id', 'timestamp']], 
            data['choice'], 
            test_size=0.2, 
            random_state=147
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size=0.25, 
            random_state=147
        )


        train = pd.DataFrame({
            "img_directory": X_train['img_path'].values,
            "img_url": X_train['image'].values,
            "annotation_id": X_train['annotation_id'].values,
            "timestamp" : X_train['timestamp'].values,
            "label": y_train.values
        }).reset_index(drop=True)
        
        val = pd.DataFrame({
            "img_directory": X_val['img_path'].values,
            "img_url": X_val['image'].values,
            "annotation_id": X_val['annotation_id'].values,
            "timestamp" : X_val['timestamp'].values,
            "label": y_val.values
        }).reset_index(drop=True)
        
        test = pd.DataFrame({
            "img_directory": X_test['img_path'].values,
            "img_url": X_test['image'].values,
            "annotation_id": X_test['annotation_id'].values,
            "timestamp" : X_val['timestamp'].values,
            "label": y_test.values
        }).reset_index(drop=True)

        if(output_csvs):
            train.to_csv(csv_output_dir + "train")
            val.to_csv(csv_output_dir + "val")
            test.to_csv(csv_output_dir + "test")
    
    return train, val, test


#Defining a dataset class to import the images. We resize them to 224 by 224 since that's what the model expects, but make no other transformations.

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts PIL to Tensor
])

class CustomImageDataset(Dataset):
    def __init__(self, data_df, transform = None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["img_directory"]
        label = int(self.data.iloc[idx]["label"])
        img_url = self.data.iloc[idx]["img_url"]
        annotation_id = str(self.data.iloc[idx]["annotation_id"])
    
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "pixel_values": image,
            "labels": label, 
            "img_path": image_path,
            "img_url": img_url,
            "annotation_id" : annotation_id
        }

class CustomEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.data = embeddings.merge(labels, left_index=True, right_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.data.filter(items = range(0, 768)).iloc[idx].to_numpy()
        label = self.data['label'].iloc[idx]

        
        return {
            "embeddings": embedding,
            "labels": label, 
        }
    
    

def get_train_val_test_dataloaders(train_df, val_df, test_df, batch_size = 32, generator = None):
    #Creating the dataset and loading it into batches with the DataLoader class
    train_dataset = CustomImageDataset(train_df, transform=transform)
    val_dataset = CustomImageDataset(val_df, transform=transform)
    test_dataset = CustomImageDataset(test_df, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, generator = generator, worker_init_fn=seed_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, generator = generator, worker_init_fn=seed_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, generator = generator, worker_init_fn=seed_worker)

    return train_dataloader, val_dataloader, test_dataloader

def embedding_to_dataloader(embeddings, labels, batch_size = 32, generator = None):
    dataset = CustomEmbeddingDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, generator = generator, worker_init_fn=seed_worker)

    return dataloader


#Save all the embeddings from a model to a ChromaDB dataset 
def save_full_embeddings(model, data, collection_name, persist_directory = "embedding_data/", device = "cuda"):
    """
    model: The encoder to encode the images with
    data: The image dataset to get embeddings from
    collection_name: The name of the ChromaDB dataset
    persistent_directory: The path of the ChromaDB dataset
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.create_collection(name=collection_name)

    for batch in data:
        images = batch['pixel_values'].to(device)
        annotation_ids = batch['annotation_id']

        embedding = model(images)

        collection.add(
            embeddings=embedding.tolist(),
            ids = annotation_ids
        )

        del embedding
        del images
        del annotation_ids

        torch.cuda.empty_cache()
        gc.collect()




#Load all the embeddings from a ChromaDB dataset 
def load_full_embeddings(original_df, collection_name, persist_directory = "embedding_data/"):
    """
    persistent_directory: The path of the ChromaDB dataset
    collection_name: The name of the ChromaDB dataset
    original_df: The name of the dataframe containing the ids, labels, and urls of the images in the dataset
    """

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)

    db_output = collection.get(ids = original_df['annotation_id'].astype(str).tolist(), include = ['embeddings'])
    embeddings = db_output['embeddings']
    labels = original_df['label']

    db_df = pd.DataFrame(embeddings)
    db_df['ids'] = db_output['ids']
    db_df['ids'] = db_df['ids'].astype('int64')

    db_df = db_df.merge(original_df, left_on = 'ids', right_on='annotation_id')

    embeddings = db_df.filter(items = range(0, 768))
    labels = db_df['label']
    img_urls = db_df['img_url']
    a_ids = db_df['annotation_id']

    return embeddings, labels, img_urls, a_ids

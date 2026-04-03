import pandas as pd
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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
from src.seeds import seed_worker

seed_value = 147

#Helper function to get annotation result from differently formatted data
def get_annotation_result(x):
    if len(x) < 1:
        return np.nan
    else:
        #When there are two annotations for the same image, we take the most recent one. 
        if len(x) > 1:
            first_result_time = pd.to_datetime(x[0]['created_at'])
            second_result_time = pd.to_datetime(x[1]['created_at'])
            if first_result_time > second_result_time:
                 annotation_data = x[0]
            else:
                 annotation_data = x[-1]
        else:
            annotation_data = x[0]


        result = annotation_data['result']
        was_cancelled = annotation_data['was_cancelled']

        #Mark as unlabeled if this annotation was cancelled
        if was_cancelled:
            return np.nan

        if len(result) < 1:
            return np.nan
        else:
            return result[0]['value']['choices'][0]


#Get the label, image URL, and annotation id of the images
def get_data_urls(labels_csv, binarize = False, include_location = False):
    """
    labels_csv: The annotation file exported from LabelStudio, in csv form
    
    """
    if 'json' in labels_csv:
        raw_data = pd.read_json(labels_csv)

        annotations = raw_data['annotations'].apply(get_annotation_result)
        images = raw_data['data'].apply(lambda x: x['image'])
        ids = raw_data['id']

        data = pd.DataFrame({'choice' : annotations, 'image' : images, 'id' : ids})

    else:
        data = pd.read_csv(labels_csv)
        if include_location:
            data = data.get(["choice", "image", "id", "location"]).dropna()
        else:
            data = data.get(["choice", "image", "id", ]).dropna()
            data['choice'] = data['choice'].astype(str)



    data["choice"] = data["choice"].astype(str).fillna("0").str.extract(r"(\d)").astype(int) - 1

    if binarize:
        data["choice"] = data["choice"].apply(lambda x: -1 if x < 0 else 1 if x > 1 else 0)

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
        ann_id = data.iloc[i]['id']
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

    return pd.DataFrame({"id" : annotations, 
                         "img_path" : img_paths})


#Gather the data for image labels and paths into one big DataFrame
def get_data(labels_csv, image_dir, replace_images = False, binarize = False, include_location = False):
    """
    labels_csv: The annotation file exported from LabelStudio, in csv form

    image_dir: The directory for the images to be saved to/gathered from

    replace_images: Whether to replace the images currently in the directory
    """
    url_data = get_data_urls(labels_csv, binarize = binarize, include_location = include_location)

    download_images(url_data, image_dir, replace_images)

    image_df = get_images_df(image_dir)

    full_data = url_data.merge(image_df, left_on = "id", right_on="id")

    if include_location:
        full_data = full_data.get(["choice", "img_path", "id", "image", "location"])
    else:
        full_data = full_data.get(["choice", "img_path", "id", "image"])
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
            data[['img_path', 'image', 'id', 'timestamp']], 
            data['choice'], 
            test_size=0.2, 
            random_state=seed_value
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size=0.25, 
            random_state=seed_value
        )


        train = pd.DataFrame({
            "img_path": X_train['img_path'].values,
            "image": X_train['image'].values,
            "id": X_train['id'].values,
            "timestamp" : X_train['timestamp'].values,
            "choice": y_train.values
        }).reset_index(drop=True)
        
        val = pd.DataFrame({
            "img_path": X_val['img_path'].values,
            "image": X_val['image'].values,
            "id": X_val['id'].values,
            "timestamp" : X_val['timestamp'].values,
            "choice": y_val.values
        }).reset_index(drop=True)
        
        test = pd.DataFrame({
            "img_path": X_test['img_path'].values,
            "image": X_test['image'].values,
            "id": X_test['id'].values,
            "timestamp" : X_test['timestamp'].values,
            "choice": y_test.values
        }).reset_index(drop=True)

        if(output_csvs):
            train.to_csv(csv_output_dir + "train")
            val.to_csv(csv_output_dir + "val")
            test.to_csv(csv_output_dir + "test")
    
    return train, val, test

#TODO: Add a function called get_k_folds that splits the data into k folds for cross validation, and outputs dataframes for each fold. Should also stratify data based on label and location
def get_k_folds(df: pd.DataFrame, strat_cols: list, k: int) -> list:
    """
    Randomly splits a DataFrame into k approximately equal sub-dataframes, 
    stratified by a list of features.
    
    Args:
        df: The pandas DataFrame to split.
        strat_cols: A list of column names to stratify by.
        k: The number of folds to generate.
        
    Returns:
        A list containing k pandas DataFrames.
    """
    # 1. Create a single composite key for stratification
    # This joins the string values of the strat_cols to treat them as a single target class
    strat_key = df[strat_cols].astype(str).agg('_'.join, axis=1)
    
    # 2. Initialize the StratifiedKFold object
    # shuffle=True ensures the data is randomized before splitting
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed_value)
    
    folds = []
    
    # 3. Generate the folds
    # skf.split generates (train_index, test_index) for each fold.
    # The 'test_index' represents the unique chunk of data for that specific fold.
    for _, test_idx in skf.split(df, strat_key):
        fold_df = df.iloc[test_idx].copy()
        folds.append(fold_df)
        
    return folds

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
        image_path = self.data.iloc[idx]["img_path"]
        label = int(self.data.iloc[idx]["choice"])
        img_url = self.data.iloc[idx]["image"]
        id = str(self.data.iloc[idx]["id"])
    
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "pixel_values": image,
            "labels": label, 
            "img_path": image_path,
            "img_url": img_url,
            "id" : id
        }
    
class InferenceDataset(Dataset):
    def __init__(self, data_df, transform = None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["img_path"]
        img_url = self.data.iloc[idx]["image"]
        id = str(self.data.iloc[idx]["id"])
    
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "pixel_values": image,
            "img_path": image_path,
            "img_url": img_url,
            "id" : id
        }

class CustomEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.data = embeddings.merge(labels, left_index=True, right_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.data.filter(items = range(0, 768)).iloc[idx].to_numpy()
        label = self.data['choice'].iloc[idx]

        
        return {
            "embeddings": embedding,
            "labels": label, 
        }
    

def get_inference_dataloader(data_df, batch_size = 32, generator = None):
    dataset = InferenceDataset(data_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, generator = generator, 
                            worker_init_fn=seed_worker)

    return dataloader

def get_train_val_test_dataloaders(train_df, val_df, test_df, batch_size = 32, generator = None):
    #Creating the dataset and loading it into batches with the DataLoader class
    train_dataset = CustomImageDataset(train_df, transform=transform)
    val_dataset = CustomImageDataset(val_df, transform=transform)
    test_dataset = CustomImageDataset(test_df, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, generator = generator, worker_init_fn=seed_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, generator = generator, worker_init_fn=seed_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, generator = generator, worker_init_fn=seed_worker)

    return train_dataloader, val_dataloader, test_dataloader

def pipe_to_dataloader(df, batch_size = 32, generator = None):
    dataset = CustomImageDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, generator = generator, worker_init_fn=seed_worker)

    return dataloader


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
    with torch.no_grad():
        for batch in data:
            images = batch['pixel_values'].to(device)
            ids = batch['id']

            embedding = model(images)

            collection.add(
                embeddings=embedding.tolist(),
                ids = ids
            )

            del embedding
            del images
            del ids

            torch.cuda.empty_cache()
            gc.collect()




#Load all the embeddings from a ChromaDB dataset 
def load_full_embeddings(original_df, collection_name, persist_directory = "embedding_data/"):
    """
    persistent_directory: The path of the ChromaDB dataset
    collection_name: The name of the ChromaDB dataset
    original_df: The name of the dataframe containing the ids, labels, and urls of the images in the dataset

    Returns: embeddings, labels, urls, and annotation ids of the images in the dataset, in that order
    """

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)

    db_output = collection.get(ids = original_df['id'].astype(str).tolist(), include = ['embeddings'])
    embeddings = db_output['embeddings']
    labels = original_df['choice']

    db_df = pd.DataFrame(embeddings)
    db_df['ids'] = db_output['ids']
    db_df['ids'] = db_df['ids'].astype('int64')

    db_df = db_df.merge(original_df, left_on = 'ids', right_on='id')

    embeddings = db_df.filter(items = range(0, 768))
    labels = db_df['choice']
    img_urls = db_df['image']
    a_ids = db_df['id']

    return embeddings, labels, img_urls, a_ids

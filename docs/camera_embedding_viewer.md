# Camera Embedding Viewer Pipeline

This document describes the steps required to prepare reduced PCA embeddings for a given camera and load them into the Embedding Visualization Tool.

---

### 1. Required Configuration

For each camera dataset, define the following variables: 

* **`reduced_embeddings`** *(np.ndarray)*: The PCA-reduced embeddings for the camera.
  Shape: `(n_samples, 5)`
  Each row represents one image embedding reduced to 5 principal components.

* **`labels`** *(pd.Series or array-like)*: the label assigned to each embedding
  Shape: `(n_samples,)`

* **`image_urls`** *(pd.Series or array-like)*: the URLs (or file paths) corresponding to each embedding image
  Shape: `(n_samples,)`
  These are used for hover image previews in the visualization tool.

* **`a_ids`** *(pd.Series or array-like)*: the annotation IDs corresponding to each embedding.
  Shape: `(n_samples,)`

* **`dataset_id`** *(string)* the unique dataset identifier
  This value populates the **camera dropdown menu** in the viewer
  **Convention:**
  Use the camera’s human-readable name (e.g., `"Mesa Grande N"`).

* **`metadata`** *(dictionary, optional)*: Any additional dataset information for tracking and reference.

  Example:

  ```python
  metadata = {
      "camera_name": "Mesa Grande N",
      "date_range": "2024-10 - 2025-10",
  }
  ```

* **`label_map`** *(dictionary)*: The mapping of label IDs to display names and colors in the visualization legend.

  **Required format:**

  ```python
  label_map = {
      0: {"name": "Normal", "color": "purple"},
      1: {"name": "Aormal", "color": "gold"},
  }
  ```
  **Requirements:**
  * Keys must match values present in `labels`
  * Every label must include both `name` and `color`

* **`output_path`** *(string)*: The file path where embedding data will be saved.
  **Default:**
  ```
  embedding_data/embeddings.json
  ```

---

### 2. Run the Function

```python
save_embedding_dataset(reduced_embeddings, 
                        labels, 
                        image_urls,
                        a_ids,
                        dataset_id, 
                        metadata, 
                        label_map,
                        output_path)
``````

---

### 3. Output File Structure

After running the export script, the embeddings file will follow this structure:

```
embedding_data/
└── embeddings.json
```

Each dataset entry inside the JSON will contain:

```
datasets[
  {
    dataset_id,
    metadata,
    label_map,
    points: [
      {
        pcs,
        label,
        image_url,
        a_id
      }
    ]
  }
]
```

---

### 4. Loading the Dataset in the Viewer

Once the dataset is exported:

1. Open the Embedding Visualization website
2. Navigate to the **Camera** dropdown
3. Select the `dataset_id`

The dashboard will automatically load:

* PCA projections
* Label color mapping
* Legend entries
* Hover image previews

---

### 5. Deleting a Dataset

To remove a dataset from the viewer, run:

```python
delete_embedding_dataset(
    dataset_id,
    json_path="embedding_data/embeddings.json"
)
```

This will:

* Remove the dataset from the JSON file
* Remove it from the viewer dropdown menu

---

### Notes

* Ensure embeddings are reduced to **exactly 5 PCs**
* Confirm all arrays have matching lengths
* Verify image URLs are accessible
* Ensure label IDs match the label map
* Repeat this process for each new camera dataset

# Camera Data Extraction Pipeline

This document describes the steps required to extract camera frames for a given camera and prepare them for import into Label Studio.

---

### 1. Required Configuration

For each camera, define the following variables before running the script:

* **`CAMERA_ID`** (string): The camera ID, matching the value that follows `id=` in the AlertCalifornia camera URL

* **`cam_name`** (string): A short, human-readable camera name. This will be used to create the output folder

* **`start_date`** (`datetime` object): Start of the time range to query

* **`end_date`** (`datetime` object): End of the time range to query

* **`daily_offset`** (dictionary or list): Daily time offsets (in seconds) used to adjust the query window. Since cameras do not always capture frames exactly at 00:00, the offset specifies the delay after midnight at which the image is taken. This value is required because the API must use the exact start and end epoch timestamps to successfully query the framelist.

    * **recommended format for long time ranges:**
      ```python
      {
        "jan24": [int1, int2, int3, ..., int31],
        "feb24": [int1, int2, int3, ..., int28],
        ...
      }
    
    * **alternative:** A flat list of daily offset values may also be used

  ```

---

### 2. Run the Extraction Script

Execute the following steps in order:

```python
# flatten the daily offset into a list (only required if daily_offset is a dictionary)
daily_offset_list = [b for month in daily_offset.values() for b in month]

# generate epoch timestamps with daily_offsets applied
epoch_times = generate_daily_epoch_times(
    start_date,
    end_date,
    daily_offset_list
)

# scrape camera frames
scrape_camera_frames(
    CAMERA_ID,
    epoch_times,
    save_dir=f"camera_data/{cam_name}/daily_framelist/"
)

# convert output to Label Studio format with PTZ metadata and downsample to hourly frames
json_to_label_studio(
    f"camera_data/{cam_name}/daily_framelist/",
    output_dir=f"camera_data/{cam_name}/LS_unfiltered/"
)

# filter frames by PTZ values to ensure consistency
filter_by_ptz(
    f"camera_data/{cam_name}/LS_unfiltered/",
    f"camera_data/{cam_name}/LS_import/"
)
```

---

### 3. Output Directory Structure

Running the script will create the following folder structure:

```
camera_data/
└── cam_name/
    ├── daily_framelist/
    ├── LS_unfiltered/
    └── LS_import/
```

* **`daily_framelist/`**
  Raw scraped camera frame data

* **`LS_unfiltered/`**
  Converted Label Studio JSON files with PTZ data, prior to filtering

* **`LS_import/`**
  Final filtered output. These files can be directly uploaded to a Label Studio project

---

### Notes

* Ensure all required variables are defined before running the script
* For multiple cameras, repeat this process with a new configuration for each camera

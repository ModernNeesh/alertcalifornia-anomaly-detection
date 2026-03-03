"""
Camera Data Extraction Pipeline

This script queries camera frames for a specified camera and date range,
processes them, and outputs Label Studio–ready JSON files.

See docs/camera_data_extraction.md for full usage instructions.
"""

import requests
import time
import json
import re
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASE_URL = "https://tools.alertcalifornia.org"
HEADERS = {"User-Agent": "dsc180 capstone project"}

def generate_daily_epoch_times(start_date, end_date, daily_offset, timezone="America/Los_Angeles"):

    """
    generates a list of epoch times for each day between start_date and end_date (inclusive)
    this is required to fetch daily framelists from the API

    args:
        start_date (datetime): start date
        end_date (datetime): end date
        daily_offset (list): a list of offset (int) in seconds to add to each day's epoch time
        timezone (str): timezone for the dates, default is "America/Los_Angeles"

    return:
        list of epoch times (int) for each day between start_date and end_date
    """

    tz = pytz.timezone(timezone)
    epoch_times = []

    for i, day in enumerate(pd.date_range(start_date, end_date)):
        dt = tz.localize(datetime(day.year, day.month, day.day))
        epoch_times.append(int(dt.timestamp()) + daily_offset[i])

    return epoch_times


def fetch_json(url, sleep_s=0.0, timeout=15):

    """
    fetches json data from a given url with optional sleep time between requests
    args:
        url (str): url to fetch json data from
        sleep_s (float): time to sleep between requests
        timeout (int): request timeout in seconds
    """

    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    
    except Exception as e:
        print(f"request failed: {url} ({e})")
        return None
    
    finally:
        if sleep_s:
            time.sleep(sleep_s)


def scrape_camera_frames(camera_id, epoch_times, save_dir):

    """
    scrapes framelist json data for a given camera_id between the specified epoch times
    saves the data to the specified save_dir

    args:
        camera_id (str): camera id to scrape
        epoch_times (list): list of epoch times to scrape between
        save_dir (str): directory to save the json files    
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("camera_id:", camera_id)
    print("save_dir:", save_dir)

    for i in range(len(epoch_times) - 1):
        start, stop = epoch_times[i], epoch_times[i + 1]
        date_label = datetime.fromtimestamp(start).strftime("%Y-%m-%d")

        url = f"{BASE_URL}/framelist.json?id={camera_id}&start={start}&stop={stop}"
        data = fetch_json(url, sleep_s=0.1)

        if data:
            out = save_dir / f"{camera_id}_{date_label}.json"
            json.dump(data, open(out, "w"))

        print(f"[{i+1}/{len(epoch_times)-1}] {camera_id} fetched {date_label}")


def select_frames_by_interval(paths, interval=3600):

    """
    selects frames from the given paths at the specified interval (in seconds)

    args:
        paths (list): list of frame paths
        interval (int): interval in seconds

    return:
        list of selected frame paths
    """

    selected = []
    prev_epoch = None

    for p in paths:
        match = re.search(r"\d{10}", p)
        if not match:
            continue

        epoch = int(match.group())
        if prev_epoch is None or epoch >= prev_epoch + interval:
            selected.append(p)
            prev_epoch = epoch

    return selected


def fetch_ptz(path):

    """
    fetches PTZ data for a given frame path
    
    args:
        path (str): frame path
    """

    url = f"{BASE_URL}/transform{path}?transform=ptz-combo"
    data = fetch_json(url, sleep_s=0.05)

    if data and "ptz" in data:
        return data["ptz"]
    

def json_to_label_studio(file_path, output_dir, interval=3600):

    """
    converts framelist json files to label studio format with PTZ data

    args:
        file_path (str): directory containing framelist json files
        output_dir (str): directory to save label studio formatted json files
        interval (int): interval in seconds to select frames

    return:
        path to the last processed output file
    """

    json_files = Path(file_path).glob("*.json")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    i = 0

    for file in json_files:

        data = json.load(open(file))
        paths = data.get("digitalpath-redis", [])

        selected = select_frames_by_interval(paths, interval)

        records = []
        for p in selected:
            ptz = fetch_ptz(p)
            if ptz:
                pan, tilt, zoom = ptz
                records.append({
                    "image": BASE_URL + p,
                    "ptz": {"pan": pan, "tilt": tilt, "zoom": zoom}
                })

        out = output_dir / f"{Path(file).stem}_LS.json"
        json.dump(records, open(out, "w"), indent=2)
        i += 1
        print(f"[{i}/{len(list(Path(file_path).glob('*.json')))}] processed {file.name}, length: {len(records)}")

    return out

def compute_median_ptz(json_files):

    """
    computes the median PTZ values from a list of label studio formatted json files

    args:
        json_files (list): list of json file paths

    return:
        median_pan (float): median pan value
        median_tilt (float): median tilt value
        median_zoom (float): median zoom value
    """
    
    pans, tilts, zooms = [], [], []
    
    for file in json_files:
        
        with open(file, 'r') as f:
            data = json.load(f)
            for r in data:
                pans.append(r['ptz']['pan'])
                tilts.append(r['ptz']['tilt'])
                zooms.append(r['ptz']['zoom'])
                
    median_pan = np.median(pans)
    median_tilt = np.median(tilts)
    median_zoom = np.median(zooms)

    return median_pan, median_tilt, median_zoom


def filter_by_ptz(file_path, output_dir, thresholds=(4, 0.01, 0.01)):

    """
    filters label studio formatted json files by PTZ values within specified thresholds
    organizes filtered data into monthly files in the output directory
    
    args:
        file_path (str): directory containing label studio formatted json files
        output_dir (str): directory to save filtered json files
        thresholds (tuple): thresholds for pan, tilt, and zoom filtering

    """

    json_files = list(Path(file_path).glob("*.json"))
    # print(len(json_files), "files found for filtering")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    median_pan, median_tilt, median_zoom = compute_median_ptz(json_files)
    pan_thresh, tilt_thresh, zoom_thresh = thresholds
    pan_range = (median_pan - pan_thresh, median_pan + pan_thresh)
    tilt_range = (median_tilt - tilt_thresh, median_tilt + tilt_thresh)
    zoom_range = (median_zoom - zoom_thresh, median_zoom + zoom_thresh)

    print(f"global ptz reference: pan={median_pan}, tilt={median_tilt}, zoom={median_zoom}")
    print(f"filtering ranges: pan={pan_range}, tilt={tilt_range}, zoom={zoom_range}")

    monthly_data = defaultdict(list)
    for file in json_files:
        match = re.search(r"(\d{4})-(\d{2})-\d{2}", file.name)
        if not match:
            continue
        year, month = match.groups()
        month_key = f"{year}{month}"
        monthly_data[month_key].append(file)

    for month_key, files in monthly_data.items():
        print("processing month:", month_key)

        combined_data = []

        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)

            # filter by ptz
            for item in data:
                pan = item['ptz']['pan']
                tilt = item['ptz']['tilt']
                zoom = item['ptz']['zoom']

                if (pan_range[0] <= pan <= pan_range[1] and
                    tilt_range[0] <= tilt <= tilt_range[1] and
                    zoom_range[0] <= zoom <= zoom_range[1]):
                    combined_data.append(item)

                # else:
                #     print(month_key, "excluded item with ptz:", item['ptz'])

        out_file = output_dir / f"{month_key}_filtered_LS.json"
        with open(out_file, 'w') as f:
            json.dump(combined_data, f, indent=2)

    print(f"done filtering, monthly data saved to {output_dir}")


def load_ptz_df(LS_dir):

    """
    load label studio formatted json files into a single df

    args:
        LS_dir (str): directory containing label studio formatted json files

    return:
        df (pd.DataFrame): dataframe containing all PTZ data
    """

    df = pd.DataFrame()
    for file in Path(LS_dir).glob('*_LS.json'):
        with open(file, 'r') as f:
            data = json.load(f)
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    # extract PTZ columns
    df['pan'] = df['ptz'].apply(lambda x: x['pan'])
    df['tilt'] = df['ptz'].apply(lambda x: x['tilt'])
    df['zoom'] = df['ptz'].apply(lambda x: x['zoom'])

    # remove ptz column
    df = df.drop(columns=['ptz'])

    return df

@echo off

REM Before running this script, ensure "coronado_hills_data.csv", "coronado_hills_full.json", and "training_set_cameras_data.csv" are in the camera_data directory.
REM Then, run this script by running "run_cvs.bat" in the terminal. You may need to type "./run_cvs.bat" if an error occurs.

python run_cross_validation.py --objective "triplet" --seed 2596
python run_cross_validation.py --objective "deepsad" --data-csv-name "coronado_hills_full.json" --seed 2596
python run_cross_validation.py --objective "hsad" --data-csv-name "coronado_hills_full.json" --seed 2596
python run_cross_validation.py --objective "final" --data-csv-name "training_set_cameras_data.csv" --seed 2596

python run_cross_validation.py --objective "triplet" --seed 3645
python run_cross_validation.py --objective "deepsad" --data-csv-name "coronado_hills_full.json" --seed 3645
python run_cross_validation.py --objective "hsad" --data-csv-name "coronado_hills_full.json" --seed 3645
python run_cross_validation.py --objective "final" --data-csv-name "training_set_cameras_data.csv" --seed 3645

python run_cross_validation.py --objective "triplet" --seed 4586
python run_cross_validation.py --objective "deepsad" --data-csv-name "coronado_hills_full.json" --seed 4586
python run_cross_validation.py --objective "hsad" --data-csv-name "coronado_hills_full.json" --seed 4586
python run_cross_validation.py --objective "final" --data-csv-name "training_set_cameras_data.csv" --seed 4586
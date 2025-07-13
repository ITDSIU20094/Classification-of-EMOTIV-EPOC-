import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Dataset path
kaggle_dataset = 'drammaraalhamadani/emotiv-epoc-3-classes'
output_dir = './kaggle_downloads'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Download and unzip all files from the Kaggle dataset (download everything at once)
print("Downloading and unzipping all files from the dataset...")
api.dataset_download_files(kaggle_dataset, path=output_dir, force=True, unzip=True)
print("All files downloaded and unzipped to:", output_dir)

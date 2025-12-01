import os
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm

# Configuration
DATASET_DIR = 'vww_dataset'
# Option 1: PennFudanPed (Fast, ~50MB, but only contains 'person' images)
PENN_FUDAN_URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
# Option 2: INRIA Person (Comprehensive, ~970MB, contains 'person' and 'background')
# Using Archive.org mirror as the original site is often down.
INRIA_URL = 'https://web.archive.org/web/20190301110434/ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar'

def download_url(url, output_path):
    print(f"Downloading {url}...")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=output_path) as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)
        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
    print(f"Downloaded to {output_path}")

def extract_file(file_path, extract_to):
    print(f"Extracting {file_path}...")
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.endswith('.tar') or file_path.endswith('.tar.gz'):
        with tarfile.open(file_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def setup_penn_fudan():
    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_path = os.path.join(DATASET_DIR, 'PennFudanPed.zip')
    
    if not os.path.exists(zip_path):
        download_url(PENN_FUDAN_URL, zip_path)
    
    extract_file(zip_path, DATASET_DIR)
    
    # Organize for VWW (Note: PennFudanPed only has persons)
    # We will create a 'person' folder. You will need to find 'not_person' images elsewhere 
    # (e.g., CIFAR-10 background classes, or random COCO images) to train a binary classifier.
    source_dir = os.path.join(DATASET_DIR, 'PennFudanPed', 'PNGImages')
    target_dir = os.path.join(DATASET_DIR, 'train', 'person')
    os.makedirs(target_dir, exist_ok=True)
    
    import shutil
    print("Organizing images...")
    for img_name in os.listdir(source_dir):
        if img_name.endswith('.png'):
            shutil.copy(os.path.join(source_dir, img_name), os.path.join(target_dir, img_name))
    
    print(f"✅ PennFudanPed setup complete. Images in {target_dir}")
    print("⚠️  WARNING: This dataset ONLY contains people. You need negative samples (backgrounds) to train the model.")

if __name__ == "__main__":
    # Defaulting to PennFudanPed as requested for the code
    setup_penn_fudan()
    
    # To download INRIA instead (which has backgrounds), uncomment below:
    # download_url(INRIA_URL, os.path.join(DATASET_DIR, 'INRIAPerson.tar'))

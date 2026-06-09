import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_manager import config_manager
from src.utils.logger import setup_logger

logger = setup_logger(quiet=True)

def check_data_exists(dest_dir: Path) -> bool:
    required_files = ["ratings.csv", "movies.csv"]
    
    for file in required_files:
        if not (dest_dir / file).exists():
            return False
    return True

def download_and_extract_movielens(dataset_url: str, destination_path: str = "./data/raw", force_download: bool = False):

    dest_dir = Path(destination_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if check_data_exists(dest_dir) and not force_download:
        logger.info("✅ Data already exists! No need to download.")
        logger.info(f"📍 Location: {dest_dir}")
        logger.info("You can start using the data immediately.")
        return dest_dir
    
    if force_download:
        logger.info("Force download enabled...")
    else:
        logger.info("Data not found. Starting download...")
    
    zip_path = dest_dir / "movielens.zip"
    
    try:
        logger.info(f"Downloading from {dataset_url}")
        urllib.request.urlretrieve(dataset_url, zip_path)
        
        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        
        zip_path.unlink()
        
        downloaded_path = dest_dir / "ml-latest-small"
        if downloaded_path.exists() and downloaded_path.is_dir():
            for file in downloaded_path.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(dest_dir / file.name))
            downloaded_path.rmdir()
            os.remove(f"{dest_dir}/README.txt")
        
        logger.info("✅ Download and extraction completed!")
        return dest_dir
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    destination = config_manager.get("data.raw_path")
    dataset_url = config_manager.get("data.url")
    
    if not destination or not dataset_url:
        logger.error("Missing configuration!")
        sys.exit(1)
    
    data_path = download_and_extract_movielens(dataset_url, destination, force_download=False)
    
    if data_path and check_data_exists(data_path):
        print("\n" + "="*50)
        print("🎬 MovieLens Dataset is ready!")
        print("="*50)
        print(f"📁 Path: {data_path}")
        print(f"📄 Files: ratings.csv, movies.csv, tags.csv, links.csv")
        print("="*50)
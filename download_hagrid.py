import os
import requests
import zipfile
from tqdm import tqdm

URL = "https://huggingface.co/datasets/testdummyvt/hagRIDv2_512px_10GB/resolve/main/yolo_format.zip?download=true"
DEST_DIR = r"D:\BragBoard-main\Face Detection\Hand Gesture"
ZIP_FILE = os.path.join(DEST_DIR, "hagrid_subsample.zip")

def download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading")
        
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        return True
    except Exception as e:
        print(f"Download Error: {e}")
        return False

def main():
    if not os.path.exists(DEST_DIR):
        print(f"Creating directory: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)
        
    print(f"Downloading HaGRID subsample to {ZIP_FILE}...")
    if download_file(URL, ZIP_FILE):
        print("\nDownload complete. Extracting...")
        try:
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                # Get list of files to show progress/info if needed
                # zip_ref.extractall(DEST_DIR)
                
                # Extract with progress
                file_list = zip_ref.namelist()
                for file in tqdm(file_list, desc="Extracting"):
                    zip_ref.extract(file, DEST_DIR)
                    
            print("Extraction complete.")
            
            # Cleanup
            print("Removing zip file...")
            os.remove(ZIP_FILE)
            print("Cleanup complete.")
            
            print("\n🎉 Setup finished! You can now run 'Soft.py'.")
        except Exception as e:
            print(f"Extraction failed: {e}")
    else:
        print("Download failed.")

if __name__ == "__main__":
    main()

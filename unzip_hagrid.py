import zipfile
import os
from tqdm import tqdm

ZIP_FILE = r"D:\BragBoard-main\Face Detection\Hand Gesture\hagrid_subsample.zip"
DEST_DIR = r"D:\BragBoard-main\Face Detection\Hand Gesture"

def unzip_file():
    if not os.path.exists(ZIP_FILE):
        print(f"❌ Error: Zip file not found at {ZIP_FILE}")
        return

    print(f"📂 Extracting {ZIP_FILE}...")
    print("   This might take a while due to the large file size (10GB+).")

    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            # Get list of files for progress bar
            file_list = zip_ref.namelist()
            
            for file in tqdm(file_list, desc="Extracting"):
                zip_ref.extract(file, DEST_DIR)
        
        print("\n✅ Extraction complete!")
        
        # Optional: Ask to delete zip? For now, we keep it just in case, or user can delete manually.
        # os.remove(ZIP_FILE) 
        
    except Exception as e:
        print(f"\n❌ Extraction Failed: {e}")

if __name__ == "__main__":
    unzip_file()

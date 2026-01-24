
import tarfile
import zipfile
import os

def setup_dataset():
    base_path = r"d:\BragBoard-main\Face Detection"
    tar_path = os.path.join(base_path, "mpii_human_pose_v1.tar.gz")
    zip_path = os.path.join(base_path, "mpii_human_pose_v1_u12_2.zip")
    
    # Extract Images
    if os.path.exists(tar_path):
        print(f"Extracting {tar_path}...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=base_path)
            print("Images extracted successfully.")
        except Exception as e:
            print(f"Error extracting tar: {e}")
    else:
        print(f"Warning: {tar_path} not found.")

    # Extract Annotations
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_path)
            print("Annotations extracted successfully.")
        except Exception as e:
            print(f"Error extracting zip: {e}")
    else:
        print(f"Warning: {zip_path} not found.")

if __name__ == "__main__":
    setup_dataset()

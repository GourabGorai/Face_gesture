
import zipfile
import os

def reextract():
    base_path = r"d:\BragBoard-main\Face Detection"
    zip_path = os.path.join(base_path, "mpii_human_pose_v1_u12_2.zip")
    
    print(f"Zip path: {zip_path}")
    if not os.path.exists(zip_path):
        print("Zip file not found!")
        return

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("Extracting...")
            # Extract first file to check path
            first = zip_ref.namelist()[0]
            print(f"First file in zip: {first}")
            
            zip_ref.extractall(base_path)
            print("Extracted all.")
            
            expected_path = os.path.join(base_path, first)
            print(f"Checking {expected_path}...")
            if os.path.exists(expected_path):
                print("Exists!")
            else:
                print("Does NOT exist!")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    reextract()

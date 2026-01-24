
from mpii_utils import MPIIDataset
import os
import glob

def test_load():
    root_dir = r"d:\BragBoard-main\Face Detection"
    sub_dir = os.path.join(root_dir, "mpii_human_pose_v1_u12_2")
    
    # search for mat file
    mat_files = glob.glob(os.path.join(sub_dir, "*.mat"))
    if not mat_files:
        print(f"No .mat files found in {sub_dir}")
        # Try finding in sub-sub dir or root?
        return

    mat_file = mat_files[0]
    print(f"Found mat file: {mat_file}")
    
    print("Initializing dataset...")
    try:
        ds = MPIIDataset(root_dir, mat_file)
        print(f"Dataset length: {len(ds)}")
        if len(ds) > 0:
            print("First item:", ds[0])
            print("Successfully loaded!")
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    test_load()


import tarfile
import os

tar_path = r"d:\BragBoard-main\Face Detection\mpii_human_pose_v1.tar.gz"

if not os.path.exists(tar_path):
    print(f"File not found: {tar_path}")
else:
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # List first 20 members to avoid spamming
            members = tar.getnames()
            print(f"Total files: {len(members)}")
            print("First 20 files:")
            for m in members[:20]:
                print(m)
            
            # Check for mat file
            mat_files = [m for m in members if m.endswith(".mat")]
            if mat_files:
                print("\nFound .mat files inside tar:")
                for m in mat_files:
                    print(m)
            else:
                print("\nNo .mat files found inside tar.")
    except Exception as e:
        print(f"Error opening tar: {e}")

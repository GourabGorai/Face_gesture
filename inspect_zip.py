
import zipfile

zip_path = r"d:\BragBoard-main\Face Detection\mpii_human_pose_v1_u12_2.zip"
try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        print("Zip contents:")
        for name in z.namelist():
            print(name)
except Exception as e:
    print(f"Error reading zip: {e}")

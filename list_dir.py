
import os

path = r"d:\BragBoard-main\Face Detection\mpii_human_pose_v1_u12_2"
print(f"Listing: {path}")
if os.path.exists(path):
    for f in os.listdir(path):
        print(f)
else:
    print("Path does not exist")

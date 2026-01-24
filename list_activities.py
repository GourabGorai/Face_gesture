
import pickle
import os

map_path = r"d:\BragBoard-main\Face Detection\activity_map.pkl"

if os.path.exists(map_path):
    try:
        with open(map_path, 'rb') as f:
            activity_map = pickle.load(f)
            
        print(f"Total Activities: {len(activity_map)}")
        # Sort by name for better readability
        sorted_activities = sorted(activity_map.keys())
        for act in sorted_activities:
            print(act)
    except Exception as e:
        print(f"Error reading map: {e}")
else:
    print("Activity map not found.")

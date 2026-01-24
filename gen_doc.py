
import pickle
import os

map_path = r"d:\BragBoard-main\Face Detection\activity_map.pkl"
doc_path = r"C:\Users\USER\.gemini\antigravity\brain\1b878c58-8d2f-42fd-9c26-21d927ef35a9\activity_list.md"

if os.path.exists(map_path):
    try:
        with open(map_path, 'rb') as f:
            activity_map = pickle.load(f)
            
        sorted_activities = sorted(activity_map.keys())
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("# List of Detectable Activities\n\n")
            f.write(f"The model has been trained to detect the following **{len(sorted_activities)}** activities:\n\n")
            for act in sorted_activities:
                f.write(f"- {act}\n")
                
        print(f"Documentation generated at {doc_path}")
            
    except Exception as e:
        print(f"Error reading map: {e}")
else:
    print("Activity map not found.")

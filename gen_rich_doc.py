
import scipy.io
import os
from PIL import Image
import re

def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")

def generate_doc():
    base_path = r"d:\BragBoard-main\Face Detection"
    # Dynamic finding of mat file to avoid path issues
    import glob
    sub_dir = os.path.join(base_path, "mpii_human_pose_v1_u12_2")
    mat_files = glob.glob(os.path.join(sub_dir, "*.mat"))
    if mat_files:
        mat_file = mat_files[0]
    else:
        # Fallback
        mat_file = os.path.join(base_path, "mpii_human_pose_v1_u12_2.mat")
        
    img_dir = os.path.join(base_path, "images")
    output_dir = os.path.join(base_path, "activity_samples")
    doc_path = os.path.join(base_path, "Activity_Documentation.md")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading annotations from {mat_file}...")
    try:
        mat = scipy.io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        release = mat['RELEASE']
        annolist = release.annolist
        
        found_activities = {} # Map activity -> image_filename
        
        if hasattr(release, 'act'):
            acts = release.act
            total = len(annolist)
            print(f"Scanning {total} annotations...")
            
            for i in range(total):
                if i >= len(acts): break
                
                try:
                    act_entry = acts[i]
                    if hasattr(act_entry, 'act_name') and isinstance(act_entry.act_name, str) and act_entry.act_name:
                        activity = act_entry.act_name
                        
                        # Filter for Subway Surfers activities
                        target_activities = ['rope skipping', 'ballroom', 'boxing', 'fencing']
                        is_target = False
                        for t in target_activities:
                            if t in activity:
                                is_target = True
                                break
                        
                        if not is_target:
                            continue

                        # If we haven't found a sample for this activity yet
                        if activity not in found_activities:
                            img_name = annolist[i].image.name
                            src_path = os.path.join(img_dir, img_name)
                            
                            if os.path.exists(src_path):
                                # Process image
                                try:
                                    with Image.open(src_path) as img:
                                        img = img.convert('RGB')
                                        # Resize thumbnail (e.g. 300px width)
                                        base_width = 300
                                        w_percent = (base_width / float(img.size[0]))
                                        h_size = int((float(img.size[1]) * float(w_percent)))
                                        img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
                                        
                                        safe_name = clean_filename(activity) + ".jpg"
                                        dst_path = os.path.join(output_dir, safe_name)
                                        img.save(dst_path)
                                        
                                        #Store relative path for markdown
                                        found_activities[activity] = f"activity_samples/{safe_name}"
                                        print(f"Found sample for: {activity}")
                                except Exception as img_err:
                                    print(f"Error processing image {img_name}: {img_err}")
                            else:
                                # Image file missing
                                pass
                except Exception as e:
                    continue
                    
        print(f"Found samples for {len(found_activities)} activities.")
        
        # Generator Markdown
        print("Generating Markdown...")
        sorted_acts = sorted(found_activities.keys())
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("# Subway Surfers Activities Documentation\n\n")
            f.write(f"This model is specialized for the following **{len(sorted_acts)}** activities used in game control:\n\n")
            f.write("| Activity Name | Sample Image |\n")
            f.write("| :--- | :--- |\n")
            
            for act in sorted_acts:
                img_rel_path = found_activities[act]
                f.write(f"| **{act}** | <img src='{img_rel_path}' width='200'> |\n")
                
        print(f"Documentation saved to {doc_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_doc()

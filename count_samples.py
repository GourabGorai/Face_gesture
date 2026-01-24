
import pickle
import os
import scipy.io

def count_samples():
    base_path = r"d:\BragBoard-main\Face Detection"
    # Dynamic finding of mat file
    import glob
    sub_dir = os.path.join(base_path, "mpii_human_pose_v1_u12_2")
    mat_files = glob.glob(os.path.join(sub_dir, "*.mat"))
    mat_file = mat_files[0] if mat_files else os.path.join(base_path, "mpii_human_pose_v1_u12_2.mat")

    print(f"Loading {mat_file}...")
    mat = scipy.io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    release = mat['RELEASE']
    annolist = release.annolist
    acts = release.act
    
    # Potential Candidates
    candidates = [
        'ski jumping', 'jumping', 'rope skipping', # UP
        'ballroom', 'curling', 'squash', 'cleaning floor', # DOWN
        'boxing, sparring', 'boxing', # LEFT
        'taichi', 'yoga', 'fencing' # RIGHT
    ]
    
    counts = {c: 0 for c in candidates}
    
    total = len(annolist)
    for i in range(total):
        if i >= len(acts): break
        try:
            act_entry = acts[i]
            if hasattr(act_entry, 'act_name') and isinstance(act_entry.act_name, str):
                name = act_entry.act_name
                # Flexible matching
                for c in candidates:
                    if c in name:
                         counts[c] += 1
        except:
            pass

    print("\nSample Counts:")
    for c, count in counts.items():
        print(f"{c}: {count}")

if __name__ == "__main__":
    count_samples()

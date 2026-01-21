import os
import shutil

# Change this path to where your folders are located
base_dir = "C:/Users/Hi Krish/Downloads/RH20T_cfg3/RH20T_cfg3"

# Walk through the directory tree
for root, dirs, files in os.walk(base_dir, topdown=False):
    print(dirs)
    for d in dirs:
        if "human" in d.lower():  # case-insensitive match
            folder_path = os.path.join(root, d)
            print(f"Deleting: {folder_path}")
            shutil.rmtree(folder_path)  # Deletes the entire folder and its contents

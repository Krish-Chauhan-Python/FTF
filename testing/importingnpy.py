import numpy as np
from dotenv import load_dotenv
import os

load_dotenv() # CHANGE THE PATH IN THE ENV FILE
db_url = os.getenv('dataset_path')
files_dataset = os.listdir(db_url)
for i in files_dataset:
    current_task = os.path.join(db_url, i)
    ls_current = os.listdir(current_task)
    print(i)
    for j in ls_current:
        current_cam = os.path.join(current_task, j)
        print(current_cam)
        try: 
            x = np.load(os.path.join(current_cam, "force_torque.npy"), allow_pickle=True).items()

            print(np.array2string(x, separator=', '))
        except Exception as e:
            print(f"Error loading file: {e}")
print(os.listdir(db_url))
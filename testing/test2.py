import numpy as np
from dotenv import load_dotenv
import os

data = np.load("C:/Altair Projects/FTF/dataset/task_0215_user_0015_scene_0010_cfg_0001/cam_043322070878/timestamps.npy", allow_pickle=True).item()
print(len(data['color']))
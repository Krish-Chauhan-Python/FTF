import os
import argparse
import random
from glob import glob

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and test task lists.")
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory containing task_* folders")
    parser.add_argument("--split", type=float, default=0.2, help="Ratio for the test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    task_dirs = sorted(glob(os.path.join(args.data_root, "task_*")))
    if not task_dirs:
        print(f"No task_* directories found in {args.data_root}")
        return
        
    random.shuffle(task_dirs)
    
    split_idx = int(len(task_dirs) * (1 - args.split))
    train_tasks = task_dirs[:split_idx]
    test_tasks = task_dirs[split_idx:]
    
    with open(os.path.join(args.data_root, "train.txt"), "w") as f:
        for t in train_tasks:
            f.write(f"{os.path.basename(t)}\n")
            
    with open(os.path.join(args.data_root, "test.txt"), "w") as f:
        for t in test_tasks:
            f.write(f"{os.path.basename(t)}\n")
            
    print(f"Split {len(task_dirs)} tasks: {len(train_tasks)} train, {len(test_tasks)} test.")

if __name__ == "__main__":
    main()

import random
import os

def collect_folders(source_dir):
    return [
        entry
        for entry in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, entry))
    ]


def split_train_test(source_dir, train_file="train.txt", test_file="test.txt", split_ratio=0.8, seed=42):
    folders = collect_folders(source_dir)

    if not folders:
        print(f"[WARN] No folders found in: {source_dir}")
        return

    random.seed(seed)
    random.shuffle(folders)

    split_idx = int(len(folders) * split_ratio)
    train = folders[:split_idx]
    test = folders[split_idx:]

    with open(train_file, "w") as f:
        f.write("\n".join(train))
    with open(test_file, "w") as f:
        f.write("\n".join(test))

    print(f"Total:  {len(folders)} folders")
    print(f"Train:  {len(train)} -> '{train_file}'")
    print(f"Test:   {len(test)} -> '{test_file}'")

# --- Usage ---
split_train_test(
    source_dir=r"C:\Altair Projects\FTF\test"
)
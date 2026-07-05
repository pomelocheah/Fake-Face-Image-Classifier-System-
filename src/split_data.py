import os
import shutil
from sklearn.model_selection import train_test_split


data_dir = "data_split"

real_dir = os.path.join(data_dir, "real")
fake_dir = os.path.join(data_dir, "fake")


real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]

files = real_files + fake_files
labels = [1] * len(real_files) + [0] * len(fake_files)


X_train, X_temp, y_train, y_temp = train_test_split(
    files,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


base_dir = "data_split"

splits = {
    "train": X_train,
    "val": X_val,
    "test": X_test
}

for split in splits:
    os.makedirs(os.path.join(base_dir, split, "real"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, "fake"), exist_ok=True)


def copy_files(file_list, split_name):
    for f in file_list:
        filename = os.path.basename(f)

        if "real" in f:
            target_path = os.path.join(base_dir, split_name, "real", filename)
        else:
            target_path = os.path.join(base_dir, split_name, "fake", filename)

        shutil.copy(f, target_path)


copy_files(X_train, "train")
copy_files(X_val, "val")
copy_files(X_test, "test")

print("Dataset split completed!")

# ======================
# 6. Count images in each split
# ======================

def count_images(folder):
    real_path = os.path.join(folder, "real")
    fake_path = os.path.join(folder, "fake")

    real_count = len(os.listdir(real_path))
    fake_count = len(os.listdir(fake_path))

    total = real_count + fake_count

    return real_count, fake_count, total


splits = ["train", "val", "test"]

print("\n===== Dataset Split Summary =====")

for s in splits:
    folder = os.path.join(base_dir, s)
    real_c, fake_c, total_c = count_images(folder)

    print(f"\n{s.upper()} SET:")
    print(f"Real images: {real_c}")
    print(f"Fake images: {fake_c}")
    print(f"Total images: {total_c}")
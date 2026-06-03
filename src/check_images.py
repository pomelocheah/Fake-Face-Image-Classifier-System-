from PIL import Image
import os

folders = [
    "data/archive-2/fake",
    "data/archive-2/real"
]

bad_images = []

for folder in folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            bad_images.append(path)

print("Bad Images:", len(bad_images))

for img in bad_images:
    print(img)
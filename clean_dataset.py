from PIL import Image
import os

def remove_bad_images(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as e:
                print("Removing:", path)
                try:
                    os.remove(path)
                except PermissionError:
                    print("⚠️ Skipping (in use):", path)

remove_bad_images("dataset")
from glob import glob
import sys
import os
from os import path
import cv2

def main():
    types = ["test", "train"]
    categories = ["apple", "banana", "orange"]
    bag = ["bag", "noBag"]

    for t in types:
        for c in categories:
            for b in bag:
                p = os.path.join(t, c, b)
                images_path = glob(p+"/*")
                for image_path in images_path:
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (224, 224))
                    cv2.imwrite(image_path, image)

if __name__ == "__main__":
    main()

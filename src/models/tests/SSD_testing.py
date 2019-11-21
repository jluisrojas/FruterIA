import tensorflow as tf
import sys
import cv2

# Regresar un directoria para poder acceder modulo de otra carpeta
sys.path.append("..")
from ops.SSD import PriorsBoxes, bbox_center_to_rect
sys.path.append("tests/")

# Script con el proposito de probar que la implementacion
# de SSD funcione de manera correcta

def main():
    print("SSD framework testing\n\n")
    test_prior_generation()

def test_prior_generation(img_path="test.jpg"):
    size = 300
    num_features = 4
    priors = PriorsBoxes(features=num_features, num_fmap=2, total_fmaps=8,
            aspect_ratios=[1,2,3,1/2,1/3], img_size=size)
    
    image = cv2.imread(img_path)
    image = cv2.resize(image, (size, size))

    # dibuja feature map
    f_size = size / num_features
    for i in range(num_features):
        for j in range(num_features):
            x = int(i*f_size)
            y = int(j*f_size)
            w = h = int(f_size)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # dibuja los priors
    for prior in priors[2, 2]:
        x, y, w, h = bbox_center_to_rect(prior)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # cv2.namedWindow('test_ssd',cv2.WINDOW_NORMAL)
    cv2.imshow("test_ssd", image)
    # cv2.resizeWindow("test_ssd", 600,600)
    cv2.waitKey(0)

main()

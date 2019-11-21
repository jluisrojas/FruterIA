import cv2
import tensorflow as tf

# Metodo que dibuja un bounding box a una image
# Args:
#   img: cv2 image
#   bbox: tensor of shape [4] (x, y, w, h)
def draw_bbox(img=None, bbox=None, color=(0, 255, 0)):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)

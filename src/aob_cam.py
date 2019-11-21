import cv2
import os
import json
import tensorflow as tf
import numpy as np
from goprocam import GoProCamera

path_to_model = "trained_models/AOB_MobileNetV2_01/"

def main():
    print("[INFO] Loading model")
    model = tf.keras.models.load_model(path_to_model+"model.h5")

    with open(path_to_model+"setup.json", "r") as data:
        setup = json.load(data)

    w, h, _ = setup["input_shape"]
    classes = setup["classes"]

    gpCam = GoProCamera.GoPro()
    cam = cv2.VideoCapture("udp://127.0.0.1:10000")

    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()

        crop_frame = frame[0:224, 104:328]

        frame_tensor = tf.convert_to_tensor(crop_frame)
        frame_tensor /= 255
        frame_tensor = tf.expand_dims(frame_tensor, 0)

        prediction = model.predict(frame_tensor)
        prediction *= 100.0
        prediction = tf.cast(prediction, tf.int8)
        #print(prediction)
        index = tf.math.argmax(prediction, axis=1)
        index = tf.keras.backend.get_value(index)[0]

        cat = classes[index]
        #print(cat)

        if cat != "orange":
            cv2.putText(crop_frame, cat , (10, 45),  cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)

        cv2.imshow("test", crop_frame)
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")

    cam.release()
    cv2.destroyAllWindows()

main()

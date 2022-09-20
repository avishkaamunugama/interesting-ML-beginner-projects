import face_recognition
from pathlib import Path
from PIL import Image
import joblib
import pickle
import numpy as np


class Prediction:
    def __init__(self, file):
        self.file = file

    def make_pred(self):

        # Load face encodings
        with open('ds_comp/encodings.dat', 'rb') as f:
            encodings = pickle.load(f)
        print(len(encodings))
        # Grab the list of names and the list of encodings
        face_names = list(encodings.keys())
        face_encodings = list(encodings.values())

        face = face_recognition.load_image_file(self.file)

        face_encoded = face_recognition.face_encodings(face)[0]

        closest_face_distance = 1.0
        closest_face_image_path = ""
        celebrity_name = ""
        for encoding in face_encodings:
            # try:
            new_image_distance = face_recognition.face_distance([encoding[0]], face_encoded)[0]
            if new_image_distance < closest_face_distance:
                closest_face_distance = new_image_distance
                closest_face_image_path = encoding[1]
                celebrity_name = encoding[2]
                print(encoding)

                print(encoding[1])
        # except IndexError:
        #     print("bad image")

        similarity = round((1.0-closest_face_distance)*100, 2)

        print(similarity)
        print(celebrity_name)
        print(closest_face_image_path)

        return [closest_face_image_path, similarity, celebrity_name]

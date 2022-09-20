import face_recognition
from pathlib import Path
import pickle
import os

data_path = Path("/home/avishka/PycharmProjects/your_famous_lookalike/static/")

encodings = {}

for img_path in data_path.glob("*.jpg"):
    name = os.path.basename(img_path)
    name = os.path.splitext(name)[0]
    print(name)
    try:
        face = face_recognition.load_image_file(img_path)  # load image
        face_encoded = face_recognition.face_encodings(face)[0]  # encodes the image
        encodings[name] = [face_encoded, img_path.name, name]
    except:
        print("bad image")

print(len(encodings))

# save to file
with open('encodings.dat', 'wb') as f:
    pickle.dump(encodings, f)

print("Successfully saved to disk.")

























# if data is arranged in folders according to the celebrity name
# celeb_name = [
#     'ben_afflek',
#     'jerry_seinfeld',
#     'elton_john',
#     'madonna',
#     'mindy_kaling'
# ]


# encodings = {}


# load 1 image from each folder and add its encoding to the list
# def load_data(celeb_name):
#     for img_path in Path("/home/avishka/PycharmProjects/your_famous_lookalike/data/data/train/" + celeb_name).glob(
#             "*.jpg"):
#         face = face_recognition.load_image_file(img_path)  # load image
#         print(img_path)
#         try:
#             face_encoded = face_recognition.face_encodings(face)[0]  # encodes the image
#             encodings[celeb_name] = [face_encoded, img_path.name]
#             break
#         except IndexError:
#             print("bad image")
#             continue
#
#
# for name in celeb_name:
#     print(name)
#     load_data(name)
#
# print(encodings.keys())

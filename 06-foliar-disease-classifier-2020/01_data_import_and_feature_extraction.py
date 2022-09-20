import shutil
from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16
import pandas as pd
from tensorflow import keras

# Path to folders with training data
data_path = Path("images")

categories = {
    0: "healthy",
    1: "multiple_diseases",
    2: "rust",
    3: "scab"
}

images = []
labels = []

# read th csv file to  a data frame
df = pd.read_csv("train.csv")


# print(df.head())


# Iterates one by one through all the images in a given folder, converts them to numpy arrays and adds them to image
# list
def load_data(image_id, value):
    for img in data_path.glob("*.jpg"):
        jpg_filename = img
        if img.name == image_id + ".jpg":
            img = image.load_img(img, target_size=(224, 224))

            # Convert the image to a numpy array
            image_array = image.img_to_array(img)

            # Add the image to the list of images
            images.append(image_array)

            # For each 'not dog' image, the expected value should be 0
            labels.append(value)


# Iterates through the data frame and assigns a label for each image
for index, row in df.iterrows():
    if row['healthy'] == 1:
        load_data(row['image_id'], 0)
    elif row['multiple_diseases'] == 1:
        load_data(row['image_id'], 1)
    elif row['rust'] == 1:
        load_data(row['image_id'], 2)
    elif row['scab'] == 1:
        load_data(row['image_id'], 3)


# Create a single numpy array with all the images we loaded
x_train = np.array(images)

# Also convert the labels to a numpy array
y_train = np.array(labels)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 4)

# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

# Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")

# Save the matching array of expected values to a file
joblib.dump(y_train, "y_train.dat")


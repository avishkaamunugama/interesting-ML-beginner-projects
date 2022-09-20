from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16

# Link to dataset used to train the model https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia


# Path to folders with training data
covid_train = Path("/content/chest_xray/train/PNEUMONIA")
normal_train = Path("/content/chest_xray/train/NORMAL")
train_images = []
train_labels = []

# Loads the images, converts each to a numpy array and adds the array to the image array variable .
# The expected label of the food category is added to the labels array variable
# training data import
count = 0
for img in normal_train.glob("*.jpeg"):
    count += 1
    if count > 1340:
        break
    # shrinks the image resolution
    img = image.load_img(img, target_size=(224, 224))
    image_array = image.img_to_array(img)
    train_images.append(image_array)
    # simultaneously adds a label to label array
    train_labels.append(0)
print(count)

count = 0
for img in covid_train.glob("*.jpeg"):
    count += 1
    if count > 1340:
        break
    # shrinks the image resolution
    img = image.load_img(img, target_size=(224, 224))
    image_array = image.img_to_array(img)
    train_images.append(image_array)
    # simultaneously adds a label to label array
    train_labels.append(1)
print(count)

# normalises the image data and converts the labels data to a binary matrix
x_train = np.array(train_images)
y_train = np.array(train_labels)
x_train = vgg16.preprocess_input(x_train)

# extracts the features of the images
feature_extractor = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x_train = feature_extractor.predict(x_train)
# x_test = feature_extractor.predict(x_test)

# Saves the extracted features into dat files
joblib.dump(x_train, "x_train.dat")
joblib.dump(y_train, "y_train.dat")

print("Saved imported data to disk.")

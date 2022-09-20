from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16

# builds the model using the saved structure and weights
f = Path("model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

# converts the input image to a normalised numpy array
img = image.load_img("/content/chest_xray/val/PNEUMONIA/person1950_bacteria_4881.jpeg", target_size=(224, 224))
image_array = image.img_to_array(img)
images = np.expand_dims(image_array, axis=0)
images = vgg16.preprocess_input(images)

# extracts the features of the new image
feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
features = feature_extraction_model.predict(images)

# Using the extracted features the model predicts what the image is
results = model.predict(features)

# prints the likelihood of the image being covid positive
single_result = results[0][0]
print("Probability being infected:  {}%".format(round(single_result * 100, 2)))
print("Probability being normal:  {}%".format(round((1 - single_result) * 100), 2))

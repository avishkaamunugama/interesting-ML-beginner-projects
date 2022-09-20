import pandas as pd
from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16
import keras.backend as K

# Load the json file that contains the model's structure
f = Path("model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("model_weights.h5")

df = pd.read_csv("test.csv")

data_path = Path("images")

ans_dict = {
    'image_id': [],
    'healthy': [],
    'multiple_diseases': [],
    'rust': [],
    'scab': []
}

for index, row in df.iterrows():
    for img in data_path.glob("*.jpg"):

        if img.name == "{}.jpg".format(row['image_id']):
            # Load an image file to test, resizing it to 64x64 pixels (as required by this model)
            img = image.load_img("{}/{}.jpg".format(data_path, row['image_id']), target_size=(224, 224))

            # Convert the image to a numpy array
            image_array = image.img_to_array(img)

            # Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
            images = np.expand_dims(image_array, axis=0)

            # Normalize the data
            images = vgg16.preprocess_input(images)

            # Use the pre-trained neural network to extract features from our test image (the same way we did to
            # train the model)
            feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            features = feature_extraction_model.predict(images)

            # Given the extracted features, make a final prediction using our own model
            results = model.predict(features)
            K.clear_session()

            single_result = results[0]

            ans_dict['image_id'].append(row['image_id'])
            ans_dict['healthy'].append(round(single_result[0], 2))
            ans_dict['multiple_diseases'].append(round(single_result[1], 2))
            ans_dict['rust'].append(round(single_result[2], 2))
            ans_dict['scab'].append(round(single_result[3], 2))

            print(row['image_id'])

df = pd.DataFrame(ans_dict)
print(df)

df.to_csv("submission.csv", index=False, header=True)

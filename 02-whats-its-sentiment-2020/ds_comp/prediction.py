from tensorflow.keras.models import model_from_json
from pathlib import Path
import tensorflow_hub as hub


class Prediction:
    def __init__(self, phrase):
        self.phrase = phrase

    def make_pred(self):
        # builds the model using the saved structure and weights
        f = Path("ds_comp/model_structure.json")
        model_structure = f.read_text()
        model = model_from_json(model_structure, custom_objects={'KerasLayer': hub.KerasLayer})
        model.load_weights("ds_comp/model_weights.h5")

        # make predicton
        sentiment = model.predict([[self.phrase]])
        sentiment = round(((sentiment[0][0]) * 100), 2)
        print(sentiment)

        return sentiment

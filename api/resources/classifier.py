import json
import numpy as np

from PIL import Image
from io import BytesIO

import utils

class MarketClassification(object):
    def __init__(self,models_path):
        self.model = utils.load_model(models_path)

    def image_loader(self, image_decod):
        """load image"""
        img = Image.open(BytesIO(image_decod))
        img = utils.preprocess_image(img)
        return img

    def predict(self, image):
        img_input = self.image_loader(image)
        trained_model = self.model
        response = utils.run_model(img_input, trained_model)
        return json.dumps(response)

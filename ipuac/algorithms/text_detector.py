import abc
import numpy as np
from keras.models import load_model
from ipuac import template

letters = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l',
           13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w',
           24: 'x', 25: 'y', 26: 'z'}


class TextDetector:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def detect_text(self, image, coordinate):
        pass


class Ocr(TextDetector):
    def __init__(self):
        self.model = load_model('../library/0815.h5')

    def detect_text(self, image, coordinate):
        crop_image = image.crop_element2(coordinate)
        predict_value = self.model.predict(crop_image.reshape(1, 784))
        letter = letters[np.argmax(predict_value) + 1]
        template.show_element_plt(crop_image, letter)
        return letters[np.argmax(predict_value) + 1]




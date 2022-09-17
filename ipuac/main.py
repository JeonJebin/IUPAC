import os
from algorithms import *
from model import Image, Objects, StructureFormula
from ipuac import template

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = '../resources/2-chloro-6-ethyl-9-fluoro-8-methyl-5-prophylundecane.jpg'
image = Image.get(path)

object_detector = Bfs()
feature_extractor = Vector()
text_detector = Ocr()

objects = Objects.create(object_detector.detect_objects(image))
body_image, body = objects.find_body(image, feature_extractor)
elements = objects.find_elements(image, text_detector)
combine_info = objects.find_combine_info(body, elements)

formula = StructureFormula.create(body_image, body, combine_info)
template.show_image('Result', formula.get_name(), image.image)
print(formula.get_name())

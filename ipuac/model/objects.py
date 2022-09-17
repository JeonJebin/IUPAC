from ipuac.algorithms import analysis
from .body import Body
from .elements import Elements
from .element import Element


class Objects:
    def __init__(self, body_coordinate, elements_coordinate):
        self.__body_coordinate = body_coordinate
        self.__elements_coordinate = elements_coordinate

    @staticmethod
    def create(best_bounding_boxes):
        max_area_index = max_area = 0
        for i, coordinate in enumerate(best_bounding_boxes):
            area = (coordinate[2] - coordinate[0]) * (coordinate[3] - coordinate[1])
            if area > max_area:
                max_area_index = i
                max_area = area

        body_coordinate = best_bounding_boxes.pop(max_area_index)
        elements_coordinate = best_bounding_boxes
        return Objects(body_coordinate, elements_coordinate)

    def find_body(self, image, feature_extractor):
        body_image = image.crop_body_image(self.__elements_coordinate, self.__body_coordinate)
        vertex = feature_extractor.find_feature_points(body_image)
        threshold = 2
        edge = analysis.find_connected_compounds(body_image, vertex, threshold)
        while len(edge) + 1 != len(vertex):
            edge = analysis.find_connected_compounds(body_image, vertex, threshold)
            threshold += 1
        return body_image, Body(vertex, edge)

    def find_elements(self, image, text_detector):
        candidates = []
        for coordinate in self.__elements_coordinate:
            alphabet = text_detector.detect_text(image, coordinate)
            element = Element.create(alphabet, coordinate)
            candidates.append(element)
        return Elements.create(candidates)

    def find_combine_info(self, body, elements):
        original_vertex = body.get_original_coordinate(self.__body_coordinate)
        combine_info = ['' for _ in range(len(original_vertex))]
        for element in elements:
            index, alphabet = element.find_nearest_vertex(original_vertex)
            combine_info[index] = alphabet
        return combine_info

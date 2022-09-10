import copy
import cv2 as cv
import numpy as np

from ipuac import template


class Image:
    def __init__(self, image, height, width):
        self.image = image
        self.height = height
        self.width = width

    @staticmethod
    def get(path):
        image = cv.imread(path)
        resize_image = cv.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        height, width, channel = resize_image.shape
        return Image(image, height, width)

    def copy(self):
        return copy.deepcopy(self.image)

    def preprocess_for_object_detect_mser(self):
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), dtype=np.float64) / 9
        filtering_image = cv.filter2D(gray_image, -1, kernel)
        inverted_image = np.invert(filtering_image)
        th, binary_image = cv.threshold(inverted_image, 50, 255, cv.THRESH_BINARY)

        titles = ['Filter', 'Invert', 'Binary']
        images = [filtering_image, inverted_image, binary_image]
        template.show_images('Object Detector(MSER) Preprocess', titles, images, 3, 'gray')

        return binary_image

    def preprocess_for_object_detect_bfs(self):
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), dtype=np.float64) / 25
        filtering_image = cv.filter2D(gray_image, -1, kernel)
        inverted_image = np.invert(filtering_image)
        th, binary_image = cv.threshold(inverted_image, 50, 255, cv.THRESH_BINARY)

        titles = ['Filter', 'Invert', 'Binary']
        images = [filtering_image, inverted_image, binary_image / 255]
        template.show_images('Object Detector(BFS) Preprocess', titles, images, 3, 'gray')

        return binary_image / 255

    def crop_body_image(self, elements_coordinate, body_coordinate):
        image = copy.deepcopy(self.image)
        for coordinate in elements_coordinate:
            image[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]] = 255
        body_image = image[body_coordinate[1]:body_coordinate[3], body_coordinate[0]:body_coordinate[2]]
        template.show_image('Body Analysis Preprocess', 'Crop Body', body_image)
        height, width, channel = body_image.shape
        return Image(body_image, height, width)

    def preprocess_for_fast(self):
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        return gray_image

    def remove_margin(self):
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        # 이미지 전처리
        inverted_image = np.invert(gray_image)
        height, width = inverted_image.shape

        crop_point_top = crop_point_bottom = crop_point_left = crop_point_right = 0

        # 가로선으로 이미지를 한 줄 씩 cropping하여 픽셀합을 구함
        for i in range(1, height):
            one_width_image = inverted_image[i, 1:width - 1]
            if sum(one_width_image) > 50:
                crop_point_top = i
                break

        for i in range(height - 1, 0, -1):
            one_width_image = inverted_image[i, 1:width - 1]
            if sum(one_width_image) > 50:
                crop_point_bottom = i
                break

        # 세로선으로 이미지를 한 줄 씩 cropping하여 픽셀합을 구함
        for i in range(1, width):
            one_height_image = inverted_image[1:height - 1, i]
            if sum(one_height_image) > 50:
                crop_point_left = i
                break

        for i in range(width - 1, 0, -1):
            one_height_image = inverted_image[1:height - 1, i]
            if sum(one_height_image) > 50:
                crop_point_right = i
                break

        margin = 5

        crop_point_top = crop_point_top - 3 if crop_point_top - 3 > 0 else 0
        crop_point_bottom = crop_point_bottom + 3 if crop_point_bottom + 3 < height else height - 1
        crop_point_left = crop_point_left - margin if crop_point_left - margin > 0 else 0
        crop_point_right = crop_point_right + margin if crop_point_right + margin < width else width - 1

        coordinate_of_structure_formula = [[crop_point_left, crop_point_top], [crop_point_right, crop_point_bottom]]

        remove_margin_image = self.image[
                              coordinate_of_structure_formula[0][1] * 2:coordinate_of_structure_formula[1][1] * 2,
                              coordinate_of_structure_formula[0][0] * 2:coordinate_of_structure_formula[1][0] * 2
                              ].copy()

        titles = ['Crop Body', 'Remove Top, Bottom Margin']
        images = [self.image, remove_margin_image]
        template.show_images('Remove Margin Before FeatureExtractor', titles, images, 2)

        height, width, channel = remove_margin_image.shape
        return Image(remove_margin_image, height, width)

    def preprocess_for_feature_point(self):
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        th, binary_image = cv.threshold(gray_image, 100, 255, cv.THRESH_BINARY)
        kernel = np.ones((8, 8), dtype=np.float64) / 64
        mean_value_image = cv.filter2D(binary_image, -1, kernel)
        th, preprocessed_image = cv.threshold(mean_value_image, 250, 255, cv.THRESH_BINARY)
        inverted_image = (255 - preprocessed_image) / 255

        titles = ['Gray', 'Binary', 'Filter', 'Binary', 'Invert']
        images = [gray_image, binary_image, mean_value_image, preprocessed_image, inverted_image]
        template.show_images('Feature Extractor(GBET) Preprocess ', titles, images, 5, 'gray')

        return inverted_image

    def preprocess_for_net_window(self):
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        kernel = np.ones((6, 6), dtype=np.float64) / 36
        mean_value_image = cv.filter2D(gray_image, -2, kernel)
        th, binary_image = cv.threshold(mean_value_image, 155, 255, cv.THRESH_BINARY)
        invert_image = (255 - binary_image) / 255

        titles = ['Gray', 'Filter', 'Binary', 'Invert']
        images = [gray_image, mean_value_image, binary_image, invert_image]
        template.show_images('NetWindow Preprocess', titles, images, 4, 'gray')

        return invert_image

    def preprocess_for_body_analysis(self, threshold):
        gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        th, binary_image = cv.threshold(gray_image, 100, 255, cv.THRESH_BINARY)
        kernel = np.ones((threshold, threshold), dtype=np.float64) / (threshold * threshold)
        mean_value_image = cv.filter2D(binary_image, -1, kernel)
        th, preprocessed_image = cv.threshold(mean_value_image, 250, 255, cv.THRESH_BINARY)

        # titles = ['Gray', 'Binary', 'Filter', 'Binary']
        # images = [gray_image, binary_image, mean_value_image, preprocessed_image]
        # template.show_images('Connected Compound Analysis Preprocess', titles, images, 5, 'gray')

        return preprocessed_image

    def crop_element(self, coordinate):
        element_crop = self.preprocess_for_element(coordinate)
        white_background = np.ones((28, 28), np.uint8) * 255

        # 이미지의 width, height
        width = coordinate[2] - coordinate[0]
        height = coordinate[3] - coordinate[1]

        ratio = 28. / max(width, height)
        if height > width:
            resize_width = int(ratio * width)
            resize_element = cv.resize(element_crop, (resize_width, 28), cv.INTER_AREA)
            tmp = 14 - int(resize_width / 2)
            white_background[0:28, tmp:tmp + resize_width] = resize_element

        else:
            resize_height = int(ratio * height)
            resize_element = cv.resize(element_crop, (28, resize_height), cv.INTER_AREA)
            tmp = 14 - int(resize_height / 2)
            white_background[tmp:tmp + resize_height, 0:28] = resize_element

        scale_element = white_background / 255.0
        black_element = 1 - scale_element

        return black_element
        # return black_element.reshape(1, 1, 28, 28)

    def crop_element2(self, coordinate):
        element_crop = self.preprocess_for_element(coordinate)

        # 이미지의 width, height
        width = coordinate[2] - coordinate[0]
        height = coordinate[3] - coordinate[1]

        square_edge=max(height, width)
        white_background = np.ones((square_edge,square_edge), np.uint8) * 255
        print(height,width)
        if height > width:
            tmp = int(square_edge/2) - int(width / 2)
            white_background[0:height, tmp:tmp + width] = element_crop
        else:
            tmp = int(square_edge/2) - int(height / 2)
            white_background[tmp:tmp + height, 0:width] = element_crop

        scale_element = cv.resize(white_background, (28, 28), interpolation=cv.INTER_AREA)
        black_element = 1 - scale_element/255.0

        return black_element
        # return black_element.reshape(1, 1, 28, 28)

    def preprocess_for_element(self, coordinate):
        crop_image = self.image[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
        gray_image = cv.cvtColor(crop_image, cv.COLOR_BGR2GRAY)
        th, element_crop = cv.threshold(gray_image, 250, 255, cv.THRESH_BINARY)
        kernel = np.ones((2, 2), np.float64) / 4.
        # return cv.filter2D(element_crop, -1, kernel)
        return gray_image

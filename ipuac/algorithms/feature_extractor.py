import abc
import cv2 as cv
import numpy as np
from numpy import dot
from numpy.linalg import norm

from . import net_window
from ipuac import template
from itertools import combinations


class FeatureExtractor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def find_feature_points(self, image):
        pass


class Fast(FeatureExtractor):
    def find_feature_points(self, original_image):
        image = original_image.preprocess_for_fast()

        # fast 특징점 검출 실행
        fast_threshold = 90
        fast = cv.FastFeatureDetector_create(fast_threshold)  # 임계값 설정
        feature_points = fast.detect(image)

        # 검출된 특징점 후보들을 담은 리스트 생성
        candidate_points = []
        for feature_point in feature_points:
            point = [int(feature_point.pt[0]), int(feature_point.pt[1])]
            candidate_points.append(point)
        candidate_points.sort(key=lambda x: x[0])

        # net_window 실행
        threshold = int(original_image.width / 50)
        feature_points = net_window.find(candidate_points, threshold)

        template.show_fast_and_net_window(original_image, candidate_points, feature_points)

        return feature_points


def normalized(vector):
    unit_vector = vector / norm(vector)
    return unit_vector


def vector_field(vector_list):
    vector_sum = np.array([0, 0])
    for vector in vector_list:
        vector_sum = vector_sum + vector
    return vector_sum


def cos_sim(vector1, vector2):
    similarity = dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    return similarity


def calculate_min_square_error(all_trace, a, b, c):
    min_value = 99999999
    min_mean_square_error = []
    for xi in combinations(all_trace[b], c):
        np_xi = np.array(xi)
        np_yi = np.array(all_trace[a])
        mean_square_error = np.sum((np_xi - np_yi) ** 2)
        if min_value > mean_square_error:
            min_value = mean_square_error
            min_mean_square_error = list(xi)
    return min_mean_square_error


def resize_coordinate(points):
    resize_points = []
    for i, (x, y) in enumerate(points):
        resize_points.append([int(x / 2), int(y / 2)])
    return resize_points


def execute_gaussian_blob_edge_trace(image, height, width):
    trace_points = []
    conv_filter = np.array([0.06136, 0.24477, 0.38774, 0.24477, 0.06136])

    threshold = 0.5
    for y in range(height):
        conv = np.array([])
        src = np.array(image[y, 0:width].copy())
        for x in range(2, width - 3):
            img_area = np.array(src[x - 2:x + 3].copy())
            conv = np.append(conv, (np.dot(img_area, conv_filter)))

        for x in range(len(conv)):
            if conv[x] < threshold:
                conv[x] = 0

        derivative_conv = []
        for x in range(1, len(conv) - 1):
            derivative_conv.append(conv[x + 1] - conv[x - 1])

        second_derivative_conv = []
        for x in range(1, len(derivative_conv) - 1):
            second_derivative_conv.append(derivative_conv[x + 1] - derivative_conv[x - 1])

        # blob의 중심점 찾기
        edge = []  # > 이거 극대점이 edge가 맞는지 다시 확인해야할 필요있음!
        for x in range(1, len(second_derivative_conv) - 1):
            if second_derivative_conv[x] > threshold and second_derivative_conv[x + 1] - second_derivative_conv[x - 1] < 0.001:
                edge.append(x)

        trace_index = []
        if len(edge) % 2 == 0:
            for x in range(0, len(edge), 2):
                if image[y, (int((edge[x] + edge[x + 1]) / 2))] != 0:
                    trace_index.append((int((edge[x] + edge[x + 1]) / 2), y))

        if len(trace_index) != 0:
            trace_points.append(trace_index)

    return trace_points


def find_paths(trace_points):
    complete_path = []
    current_path = [trace_points[0]]
    for y in range(len(trace_points) - 2):
        # 검출된 포인트 개수가 다를 때
        if len(trace_points[y + 1]) != len(trace_points[y]):
            # 포인트가 없을 때, 급격한 변화가 있을 때
            if len(trace_points[y]) == len(trace_points[y + 2]):
                continue

            # 새로 생긴 꼭짓점이 어디인지 확인하기 > 비용함수를 정의하여 확인하자
            gap = len(trace_points[y + 1]) - len(trace_points[y])
            # 갈래의 수가 많아지는 경우
            if gap > 0:
                min_arr = calculate_min_square_error(trace_points, y, y + 1, len(trace_points[y + 1]) - gap)
                for index, corner_candidate in enumerate(trace_points[y + 1]):
                    if corner_candidate in min_arr:
                        current_path[index].append(corner_candidate)
                    else:
                        current_path.insert(index, [corner_candidate])

            # 갈래의 수가 적어지는 경우
            else:
                min_arr = calculate_min_square_error(trace_points, y + 1, y, len(trace_points[y]) + gap)
                check = [False for i in range(len(trace_points[y]))]
                for index, corner_candidate in enumerate(trace_points[y]):
                    if corner_candidate in min_arr:
                        current_path[index].append(corner_candidate)
                        check[index] = True
                for index, ch in reversed(list(enumerate(check))):
                    if not ch:
                        complete_path.append(current_path[index])
                        current_path.pop(index)

        else:
            for index, corner_candidate in enumerate(trace_points[y + 1]):
                current_path[index].append(corner_candidate)

    complete_path.extend(current_path)

    return complete_path


def execute_vector_momentum_vertex_detection(image, trace_points):
    vertex_candidates = []

    paths = find_paths(trace_points)

    for path in paths:
        vertex_candidates.append(path[0])
        vertex_candidates.append(path[-1])

    path_grid = []
    grid = 3
    for path in paths:
        path_grid.append([])
        for i in range(0, len(path), grid):
            path_grid[-1].append(path[i])

    for path in path_grid:
        if len(path) <= 1:
            continue

        vector1_start = np.array(path.pop(0))
        vector2_start = np.array(path.pop(0))
        vector_list = [normalized(vector2_start - vector1_start)]
        for point in range(len(path) - 1):
            vector = normalized(np.array(path[point + 1]) - path[point])
            if cos_sim(vector, vector_field(vector_list)) > 0.8:
                vector_list.append(vector)
            else:
                vertex_candidates.append(path[point])
                vector_list.clear()
                vector_list.append(vector)

    template.show_find_paths(image, paths, path_grid, vertex_candidates)

    return vertex_candidates


class Vector(FeatureExtractor):
    def find_feature_points(self, original_image):
        remove_margin_image = original_image.remove_margin()

        image = remove_margin_image.preprocess_for_feature_point()
        height, width = image.shape

        trace_points = execute_gaussian_blob_edge_trace(image, height, width)
        template.show_find_trace_points(original_image, trace_points)

        candidate_points = execute_vector_momentum_vertex_detection(original_image, trace_points)

        feature_point = net_window.find_feature_point(remove_margin_image, candidate_points)
        template.show_net_window(original_image, candidate_points, feature_point)

        return feature_point

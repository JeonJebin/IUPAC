import math
import copy

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

FONT = cv.FONT_HERSHEY_PLAIN
LINE_THICKNESS = 15

# def size(input_image, set_size):
#     # 이미지, 설정 하고자 하는 크기
#     # 색 채널에 관계없이 이미지의 크기를 측정
#     if len(input_image.shape) == 3:
#         image_for_size = cv.cvtColor(input_image.copy(), cv.COLOR_BGR2GRAY)
#     else:
#         image_for_size = input_image.copy()
#     input_height, input_width = image_for_size.shape
#
#     image = input_image.copy()
#
#     if input_width > input_height:
#         ratio = set_size / input_width
#         if ratio > 1:
#             resized_image = cv.resize(image, (set_size, int(input_height / ratio)))
#         else:
#             resized_image = cv.resize(image, (set_size, int(input_height * ratio)))
#     else:
#         ratio = set_size / input_height
#         if ratio > 1:
#             resized_image = cv.resize(image, (int(input_width / ratio), set_size))
#         else:
#             resized_image = cv.resize(image, (int(input_width * ratio), set_size))
#     return resized_image


def show_image(figure, title, image, color='color'):
    plt.figure(figure)
    plt.suptitle(figure)
    plt.title(title)
    if color == 'color':
        plt.imshow(cv.cvtColor(image.astype(np.uint8), cv.COLOR_BGR2RGB))
    else:
        plt.imshow(image, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


def show_images(figure, titles, images, width, color='color'):
    height = math.ceil(len(images) / width)
    plt.figure(figure, figsize=(width * 5, height * 5))
    plt.suptitle(figure)
    for i in range(len(images)):
        plt.subplot(height, width, i + 1)
        plt.title(titles[i])
        if color == 'color':
            plt.imshow(cv.cvtColor(images[i].astype(np.uint8), cv.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], 'gray')
        plt.xticks([]), plt.yticks([])
    plt.show()


def show_detect_objects(original_image, bounding_boxes):
    detect_object_image = original_image.copy()
    for box in bounding_boxes:
        detect_object_image = cv.rectangle(detect_object_image, (box[0], box[1]), (box[2], box[3]), GREEN, 5)

    show_image('Object Detector', 'Detect Objects', detect_object_image, 'gray')


def show_fast_and_net_window(original_image, candidate_points, feature_points):
    candidate_image = original_image.copy()
    for x, y in candidate_points:
        cv.circle(candidate_image, (x, y), LINE_THICKNESS, GREEN, 2)

    result_image = original_image.copy()
    for x, y in feature_points:
        cv.circle(result_image, (x, y), LINE_THICKNESS, RED, 2)

    titles = ['Fast', 'NetWindow']
    images = [candidate_image, result_image]
    show_images('Feature Extractor(Fast) & NetWindow', titles, images, 2)


def show_find_trace_points(image, trace_points):
    trace_image = image.copy()
    for line in trace_points:
        for point in line:
            cv.circle(trace_image, point, LINE_THICKNESS, BLUE, 1, cv.LINE_AA)
    show_image('', 'Trace point using Convolution and derivation', trace_image)


def show_find_paths(image, paths, path_grid, vertax_candidates):
    titles = []
    images = []

    all_path_image = image.copy()
    for index, path in enumerate(paths):
        path_image = image.copy()
        for point in path:
            cv.circle(path_image, point, LINE_THICKNESS, BLUE, 1, cv.LINE_AA)
            cv.circle(all_path_image, point, LINE_THICKNESS, BLUE, 1, cv.LINE_AA)
        cv.circle(path_image, path[0], LINE_THICKNESS, RED, -1, cv.LINE_AA)
        cv.circle(path_image, path[-1], LINE_THICKNESS, RED, -1, cv.LINE_AA)
        cv.circle(all_path_image, path[0], LINE_THICKNESS, RED, -1, cv.LINE_AA)
        cv.circle(all_path_image, path[-1], LINE_THICKNESS, RED, -1, cv.LINE_AA)
        titles.append("Path %d" % index)
        images.append(path_image)

    show_images('Vector Momentum Vertex Detection(by path)', titles, images, 4)
    show_images('Vector Momentum Vertex Detection(all)', ['Original', 'All Path'], [image.image, all_path_image], 2)

    path_grid_image = image.copy()
    for index, path in enumerate(path_grid):
        for point in path:
            cv.circle(path_grid_image, point, LINE_THICKNESS, BLUE, 1, cv.LINE_AA)

    for point in vertax_candidates:
        cv.circle(path_grid_image, point, LINE_THICKNESS, RED, -1, cv.LINE_AA)

    show_images('Path Grid', ['All Path', 'Path Grid'], [all_path_image, path_grid_image], 2)


def show_net_window_bfs_map(map, image, candidates, group):
    height = len(map)
    width = len(map[0])
    map_image = copy.deepcopy(map)
    for i in range(0, height):
        for j in range(0, width):
            if map_image[i][j] == 0:
                map_image[i][j] = 255
            else:
                map_image[i][j] = 0

    titles = ['NetWindow Preprocess']
    images = [np.array(map_image)]

    for index, points in enumerate(group):
        group_image = image.copy()
        for point in points:
            cv.circle(group_image, candidates[point - 1], LINE_THICKNESS, BLUE, -1, cv.LINE_AA)
        titles.append('BFS Group%d' % index)
        images.append(group_image)

    show_images('NetWindow Preprocess and BFS', titles, images, 4, 'color')


def show_net_window(original_image, candidate_points, feature_points):
    candidate_image = original_image.copy()
    for x, y in candidate_points:
        cv.circle(candidate_image, (x, y), LINE_THICKNESS, GREEN, 2)

    result_image = original_image.copy()
    for x, y in feature_points:
        cv.circle(result_image, (x, y), LINE_THICKNESS, RED, 2)

    titles = ['Before', 'After']
    images = [candidate_image, result_image]
    show_images('NetWindow', titles, images, 2)


def show_connected_compounds(image, preprocess_image, vertex, edge, threshold):
    edge_image = image.copy()
    for (x, y) in edge:
        cv.line(edge_image, vertex[x], vertex[y], RED, 5, cv.LINE_AA)

    for i, x in enumerate(vertex):
        cv.circle(edge_image, x, LINE_THICKNESS, BLUE, 2, cv.LINE_AA)

    titles = ['Preprocess', 'threshold = %d' % threshold]
    images = [preprocess_image, edge_image]
    show_images('Connected Compounds Analysis', titles, images, 2)


def show_element_plt(element_image, element_alphabet):
    # plt.figure(figure)
    # plt.suptitle(figure)
    # plt.title(title)
    # if color == 'color':
    #     plt.imshow(cv.cvtColor(image.astype(np.uint8), cv.COLOR_BGR2RGB))
    # else:
    #     plt.imshow(image, 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    plt.imshow(element_image)
    plt.title('predict_letter : %s' % element_alphabet, fontsize=20)
    plt.show()

    # show_image('Element Analysis', 'predict_letter : %s' % element_alphabet, element_image)

import copy
import numpy as np
import cv2 as cv
import math

from ipuac import template


def find(keypoint_candidates, threshold):
    key_points = []
    delete = []

    keypoint_candidates.sort(key=lambda x: x[0])

    for keypoint_candidate in keypoint_candidates:
        if keypoint_candidate in delete:
            continue

        key_points.append(keypoint_candidate)

        x = keypoint_candidate[0]
        y = keypoint_candidate[1]

        for check in keypoint_candidates:
            if x + threshold > check[0] and y + threshold > check[1] > y - threshold:
                delete.append(check)

    return key_points


def count_point_in_linear_equation(image, a, b):
    sum = 0

    show = copy.deepcopy(image)

    cv.circle(show, a, 5, (255, 0, 0), 1, cv.LINE_AA)
    cv.circle(show, b, 5, (255, 0, 0), 1, cv.LINE_AA)

    if b[0] - a[0] == 0:
        x = a[0]
        for y in range(a[1], b[1]):
            sum += image[y][x]
            cv.circle(show, (x, y), 5, (0, 255, 0), 1, cv.LINE_AA)
    elif b[1] - a[1] == 0:
        y = a[1]
        for x in range(a[0], b[0]):
            sum += image[y][x]
            cv.circle(show, (x, y), 5, (0, 255, 0), 1, cv.LINE_AA)
    else:
        m = (b[1] - a[1]) / (b[0] - a[0])
        n = a[1] - m * a[0]
        for x in range(a[0], b[0]):
            y = int(m * x + n)
            sum += image[y][x]
            cv.circle(show, (x, y), 5, (0, 255, 0), 1, cv.LINE_AA)
    # cv.imshow('image', show)
    # cv.waitKey(0)

    return sum


def find_distance(a, b):
    return math.sqrt((math.pow(b[1] - a[1], 2) + math.pow(b[0] - a[0], 2)))


def create_bfs_map(keypoint_candidates, threshold, height, width):
    map = [[0 for i in range(width + threshold * 2)] for j in range(height + threshold * 2)]

    for index, keypoint_candidate in enumerate(keypoint_candidates, 1):
        x = keypoint_candidate[0] + threshold
        y = keypoint_candidate[1] + threshold

        for a in range(y - threshold, y + threshold):
            for b in range(x - threshold, x + threshold):
                map[a][b] = index

    return map


def bfs(map, start, visited):
    group = set()

    x, y = start
    queue = [start]
    visited[x][y] = True

    while queue:
        x, y = queue.pop()
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if not visited[nx][ny] and map[nx][ny] != 0:
                queue.append((nx, ny))
                visited[nx][ny] = True
                group.add(map[nx][ny])

    group_list = list(group)
    group_list.sort()
    return group_list


def find_feature_point(original_image, candidates):
    candidates.sort(key=lambda x: x[0])

    image = original_image.preprocess_for_net_window()
    height, width = image.shape
    threshold = int(width / 55)

    map = create_bfs_map(candidates, threshold, height, width)

    group = []
    visited = [[False for i in range(width + threshold * 2)] for j in range(height + threshold * 2)]
    for i in range(0, height + threshold * 2):
        for j in range(0, width + threshold * 2):
            if not visited[i][j] and map[i][j] != 0:
                group.append(bfs(map, (i, j), visited))

    template.show_net_window_bfs_map(map, original_image, candidates, group)

    feature_points = []
    for keypoint_candidate in group:
        if len(keypoint_candidate) == 1:
            feature_points.append(candidates[keypoint_candidate[0] - 1])

        elif len(keypoint_candidate) == 2:
            a = candidates[keypoint_candidate[0] - 1]
            b = candidates[keypoint_candidate[1] - 1]
            feature_points.append([int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2)])

        else:
            max_sum = -1
            max_point = -1
            for target in keypoint_candidate:
                distance_sum = pixel_sum = 0
                target_point = candidates[target - 1]
                for compare in keypoint_candidate:
                    # 직선 방정식 내부에 있는 점 확인
                    compare_point = candidates[compare - 1]
                    if compare < target:
                        pixel_sum += count_point_in_linear_equation(image, compare_point, target_point)
                    else:
                        pixel_sum += count_point_in_linear_equation(image, target_point, compare_point)
                    distance_sum += find_distance(target_point, compare_point)

                sum = pixel_sum / distance_sum
                if max_sum < sum:
                    max_sum = sum
                    max_point = target_point

            if max_point != -1:
                feature_points.append(max_point)

    return feature_points

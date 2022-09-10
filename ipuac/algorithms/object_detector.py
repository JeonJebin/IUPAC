import abc
import cv2 as cv
from collections import deque

from ipuac import template


class ObjectDetector:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def detect_objects(self, image):
        pass


class Mser(ObjectDetector):
    def detect_objects(self, original_image):
        image = original_image.preprocess_for_object_detect_mser()

        mser = cv.MSER_create()
        regions, _ = mser.detectRegions(image)
        hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        remove = []

        for i, c1 in enumerate(hulls):
            x, y, w, h = cv.boundingRect(c1)
            r1_start = (x, y)
            r1_end = (x + w, y + h)
            for j, c2 in enumerate(hulls):
                if i == j:
                    continue
                x, y, w, h = cv.boundingRect(c2)
                r2_start = (x, y)
                r2_end = (x + w, y + h)
                if r1_start[0] > r2_start[0] and r1_start[1] > r2_start[1] and r1_end[0] < r2_end[0] and r1_end[1] < \
                        r2_end[1]:
                    remove.append(i)

        bounding_boxes = []  # 검출된 box를 담을 list 생성
        # 검출된 box표시
        for j, cnt in enumerate(hulls):
            if j in remove:
                continue

            x, y, w, h = cv.boundingRect(cnt)
            margin = 1
            # 사각형 box의 좌상단, 우하단 꼭짓점을 좌표로 저장
            # 순서대로 좌상단x, 좌상단y, 우하단x, 우하단y
            bounding_boxes.append([x - margin, y - margin, x + w + margin, y + h + margin])

        bounding_boxes.sort(key=lambda x: x[0])
        template.show_detect_objects(original_image, bounding_boxes)

        return bounding_boxes


def bfs_with_area(image, start, visited, width, height):
    queue = deque([start])
    x_min, x_max = (width + 1, -1)
    y_min, y_max = (height + 1, -1)
    start_x, start_y = start
    visited[start_x][start_y] = True
    area = 0
    while queue:
        (y, x) = queue.popleft()
        # 상하좌우 검색
        for new_y, new_x in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
            if 0 > new_y or new_y >= height or 0 > new_x or new_x >= width:
                continue
            if not visited[new_y][new_x]:
                # 같은 종류의 타일인지 확인
                if image[y][x] == image[new_y][new_x]:
                    area += 1
                    visited[new_y][new_x] = True
                    queue.append((new_y, new_x))
                    x_min = min(new_x, x_min)
                    x_max = max(new_x, x_max)
                    y_min = min(new_y, y_min)
                    y_max = max(new_y, y_max)
    if area < (width / 100) * (height / 100):
        return True, []
    return False, [x_min - 2, y_min - 2, x_max + 2, y_max + 2]


class Bfs(ObjectDetector):
    def detect_objects(self, original_image):
        image = original_image.preprocess_for_object_detect_bfs()
        height, width = image.shape

        visited = [[False for _ in range(width)] for _ in range(height)]
        bounding_boxes = []

        for i in range(0, width):
            for j in range(0, height):
                if image[j][i] == 1 and not visited[j][i]:
                    isNoise, bounding_box = bfs_with_area(image, (j, i), visited, width, height)
                    if not isNoise:
                        bounding_boxes.append(bounding_box)

        template.show_detect_objects(original_image, bounding_boxes)

        return bounding_boxes

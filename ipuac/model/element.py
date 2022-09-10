class Element:
    def __init__(self, alphabet, coordinate):
        self.__alphabet = alphabet
        self.__coordinate = coordinate

    def __add__(self, other):
        alphabet = other.__alphabet + self.__alphabet
        x = int((other.__coordinate[0] + self.__coordinate[0]) / 2)
        y = int((other.__coordinate[1] + self.__coordinate[1]) / 2)
        return Element(alphabet, (x, y))

    def __str__(self):
        return self.__alphabet

    @staticmethod
    def create(alphabet, coordinate):
        x = int((coordinate[0] + coordinate[2]) / 2)
        y = int((coordinate[1] + coordinate[3]) / 2)
        return Element(alphabet, (x, y))

    def included(self, array):
        return self.__alphabet in array

    def find_nearest_vertex(self, vertex):
        minimum = 1000
        minimum_index = 0
        for i, coordinate in enumerate(vertex):
            distance = int(abs(coordinate[0] - self.__coordinate[0]) + abs(coordinate[1] - self.__coordinate[1]))
            if distance < minimum:
                minimum = distance
                minimum_index = i
        return minimum_index, self.__alphabet

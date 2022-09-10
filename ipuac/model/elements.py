class Elements:
    def __init__(self, elements):
        self.__elements = elements

    @staticmethod
    def create(element_candidates):
        combine_alphabets = {'c': 'l', 'b': 'r'}
        stack = []
        elements = []
        for candidate in element_candidates:
            if candidate.included(combine_alphabets.keys()):
                stack.append(candidate)
            elif len(stack) != 0 and str(candidate) == combine_alphabets.get(str(stack[-1])):
                elements.append(candidate + stack.pop())
            else:
                elements.append(candidate)

        return elements

    def find_combine_info(self, vertex):
        combine_info = ['' for _ in range(len(vertex))]
        for element in self.__elements:
            index, alphabet = element.find_nearest_vertex(vertex)
            combine_info[index] = alphabet
        return combine_info



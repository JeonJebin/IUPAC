from ipuac import template
from ipuac.algorithms.search import dfs_with_path, bfs
from .nomenclatures import Alkyl, Halogen
from .parent_chain import ParentChain


class Body:
    def __init__(self, vertex, edge):
        self.__vertex_count = len(vertex)
        self.__vertex_coordinates = vertex
        self.__edge = edge
        self.__graph = self.__make_graph()

    def find_parent_chain(self, image, combine_info):
        first_dfs_path = self.__find_longest_path(self.__graph, 0, combine_info)
        second_dfs_path = self.__find_longest_path(self.__graph, first_dfs_path[1][-1], combine_info)
        template.show_parent_chain(image, self.__vertex_coordinates, first_dfs_path, second_dfs_path, combine_info)
        return ParentChain(second_dfs_path)

    def get_original_coordinate(self, coordinates):
        original_coordinates = []
        for point in self.__vertex_coordinates:
            original_coordinates.append([point[0] + coordinates[0], point[1] + coordinates[1]])
        return original_coordinates

    def __make_graph(self):
        graph = [[] for _ in range(self.__vertex_count)]
        for i in range(0, len(self.__edge)):
            a = self.__edge[i][0]
            b = self.__edge[i][1]
            graph[a].append(b)
            graph[b].append(a)
        return graph

    def __find_longest_path(self, graph, start, visited_vertex):
        visited = [False for _ in range(self.__vertex_count)]
        for i, vertex in enumerate(visited_vertex):
            if not vertex:
                pass
            else:
                visited[i] = True
        all_path = []
        dfs_with_path(graph, start, visited, [], all_path)
        all_path.sort(key=lambda x: -x[0])
        return all_path.pop(0)

    def get_functional_group_candidates(self, path, combine_info):
        visited = [False for i in range(self.__vertex_count)]

        for i in path:
            visited[i] = True

        candidates = []

        # 작용기 번호 방향 정하기
        sum_left = sum_right = 0
        for i, coordinate in enumerate(path):
            for j in self.__graph[coordinate]:
                if not visited[j]:
                    sum_left += len(path) - i
                    sum_right += i + 1

        if sum_left < sum_right:
            path.reverse()

        # 번호 - 작용기 이름 찾기
        for i, coordinate in enumerate(path):
            for j in self.__graph[coordinate]:
                if not visited[j]:
                    compare = bfs(self.__graph, j, visited)
                    name = ''
                    if not combine_info[j]:
                        name = Alkyl.get_name(compare)
                    else:
                        name = Halogen.get_name(combine_info[j])
                    candidates.append((i + 1, name))

        return candidates

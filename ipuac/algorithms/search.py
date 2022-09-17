import copy
from collections import deque


def dfs_with_path(graph, start, visited, stack, all_path):
    visited[start] = True
    stack.append(start)

    has_child = False
    for i in graph[start]:
        if not visited[i]:
            visited[i] = has_child = True
            dfs_with_path(graph, i, visited, stack, all_path)
    if not has_child:
        path = copy.deepcopy(stack)
        all_path.append([len(path), path])
    stack.pop()


def bfs(graph, start, visited):
    vertex_count = 0
    seperate_point = 0
    branch_num = 0

    queue = deque([start])
    # 현재 노드를 방문 처리
    visited[start] = True
    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 하나의 원소를 뽑아 출력
        v = queue.popleft()
        vertex_count = vertex_count + 1
        # 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
        length = 0

        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
                length = length + 1

        if length > 1:
            seperate_point = vertex_count
            branch_num = length

    return vertex_count, seperate_point, branch_num

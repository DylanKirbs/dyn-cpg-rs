class FiFoQueue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue) == 0


def graph_traverse(graph, start):
    q = FiFoQueue()
    visited = set()

    def gen_graph_search(s):
        q.enqueue((None, s))
        while not q.is_empty():
            (parent, node) = q.dequeue()
            if node not in visited:
                visited.add(node)
                node.parent = parent
                for neighbour in node.children:
                    q.enqueue((node, neighbour))

    return gen_graph_search(start)


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None

    def __repr__(self):
        return self.value + " -> " + str([c.value for c in self.children])


graph = [
    Node("A"),
    Node("B"),
    Node("C"),
]

graph[0].children = [graph[1], graph[2]]

graph_traverse(graph, graph[0])
print(graph)

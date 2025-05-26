from numpy import sqrt
from Astar.Point import Point


class Graph:
    def __init__(self, adjac_lis):
        self.adjac_lis = adjac_lis

    def get_neighbors(self, v):
        return self.adjac_lis[v]

    # This is heuristic function which is having equal values for all nodes
    @staticmethod
    def euclidean_heuristic(p1: Point, p2: Point) -> float:
        """Returns length (float) between two points\n
        p1: Point - start point\n
        p2: Point - end point
        """
        return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


    def a_star_algorithm(self, start: Point, stop: Point):
        """Finds the shortest path from start to stop in graph"""
        open_lst = {start}
        closed_lst = set([])

        # poo has present distances from start to all other nodes
        # the default value is +infinity
        poo = {start: 0}

        # par contains an adjacent mapping of all nodes
        par = {start: start}

        while len(open_lst) > 0:
            current_node = None

            # it will find a node with the lowest value of f() -
            for v in open_lst:
                if current_node is None or poo[v] + self.euclidean_heuristic(v, stop) < poo[current_node] + self.euclidean_heuristic(current_node, stop):
                    current_node = v

            if current_node is None:
                print('Path does not exist!')
                return None

            # if the current node is the stop
            # then we start again from start
            if current_node == stop:
                reconst_path = []

                while par[current_node] != current_node:
                    reconst_path.append(current_node)
                    current_node = par[current_node]

                reconst_path.append(start)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all the neighbors of the current node do
            for (m, weight) in self.get_neighbors(current_node):
                # if the current node is not presenting both open_lst and closed_lst
                # add it to open_lst and note current_node as it's par
                if m not in open_lst and m not in closed_lst:
                    open_lst.add(m)
                    par[m] = current_node
                    poo[m] = poo[current_node] + weight

                # otherwise, check if it's quicker to first visit current_node, then m
                # and if it is, update par data and poo data
                # and if the node was in the closed_lst, move it to open_lst
                else:
                    if poo[m] > poo[current_node] + weight:
                        poo[m] = poo[current_node] + weight
                        par[m] = current_node

                        if m in closed_lst:
                            closed_lst.remove(m)
                            open_lst.add(m)
            open_lst.remove(current_node)
            closed_lst.add(current_node)

        print('Path does not exist!')
        return None

import numpy as np
import copy


class Graph:

    def __init__(self, adjacency_list, patches_filled, c):
        self.nodes = patches_filled
        self.adj = adjacency_list
        self.c = c


    def cost(self):
        """The cost of the graph is defined as the number of edges."""
        return (self.adj >= 0).sum() // 2


    def mst(self):
        """Compute the size of a minimum spanning tree between the food sources,
        NOT considering the actual network!"""
        # TODO check if the graph is fully connected

        def dist(food1, food2):
            # Compute the manhatten distance between two food sources. Note, if
            # both are in horizontal or vertical alignment and this alignment
            # coincides with a wall, then we need to go around the wall, so we
            # add 2 to the distance.

            manhatten_distance = np.sum(np.abs(food1 - food2))

        foods = copy.deepcopy(self.c.foods_unscaled)
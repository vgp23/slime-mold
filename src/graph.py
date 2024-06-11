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
            if (
                (food1[0] == food2[0] and food1[0] % 2 == 1) or
                (food1[1] == food2[1] and food1[1] % 2 == 1)
            ):
                manhatten_distance += 2
            return manhatten_distance

        foods = copy.deepcopy(self.c.foods_unscaled)
        print(foods)

        # keep track of which food sources we have already added to the MST
        foods_out = list(range(1, len(foods)))
        foods_in = [0]
        total_size = 0

        while len(foods_out) > 0:
            # find the minimum length edge that connects a node in food_out to food_in
            min_food = -1  # put the food index here
            min_dist = np.inf

            for food_out in foods_out:
                for food_in in foods_in:
                    if (d := dist(foods[food_out], foods[food_in])) < min_dist:
                        min_food = food_out
                        min_dist = d

            total_size += min_dist

            foods_out.remove(min_food)
            foods_in.append(min_food)

        return total_size
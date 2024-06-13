import numpy as np
import copy


class Graph:

    def __init__(self, adjacency_list, patches_filled, c):
        self.adj = adjacency_list   # neighbours for each node, -1 means no neighbour
        self.mask = patches_filled  # mask to determine if a node is considered

        # mask with only the nodes that actually have an edge
        self.actual_mask = np.array([
            any([neighbour != -1 for neighbour in neighbours]) for neighbours in self.adj])
        self.nodes = list(np.where(self.actual_mask)[0])  # indices of nodes in adj list

        self.c = c
        self.connected = self.fullyconnected()


    def coord_to_node(self, coord):
        """Convert a coordinate (of food) to the node index in the adjacency list."""
        for index, c in enumerate(self.c.all_coordinates_unscaled):
            if np.array_equal(coord, c):
                return index
        raise LookupError


    def food_nodes(self):
        """Give a list of all adjacency list indices corresponding to food sources."""
        return list(map(self.coord_to_node, self.c.foods_unscaled))


    def neighbours(self, node):
        return self.adj[node][self.adj[node] != -1]


    def cost(self):
        """The cost of the graph is defined as the number of edges."""
        return (self.adj >= 0).sum() // 2


    def fullyconnected(self, only_foods=False):
        """Check if the graph is fully connected."""
        nodes_out = self.food_nodes() if only_foods else copy.deepcopy(self.nodes)
        visited = []
        canvisit = [nodes_out.pop()]

        while len(nodes_out) > 0 and len(canvisit) != 0:
            node = canvisit.pop()
            visited.append(node)

            for neighbour in self.neighbours(node):
                if not neighbour in canvisit and not neighbour in visited:
                    canvisit.append(neighbour)
                    if neighbour in nodes_out:
                        nodes_out.remove(neighbour)

        return len(nodes_out) == 0


    def mst(self):
        """Compute the size of a minimum spanning tree between the food sources,
        NOT considering the actual network!"""

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


    def edges(self):
        """Return a list of all edges in the graph as [(node1, node2)]"""
        edges = set()

        for node in range(len(self.adj)):
            neighbours = self.neighbours(node)

            for neighbour in neighbours:
                edge = tuple(sorted([node, neighbour]))

                if edge not in edges:
                    edges.add(edge)

        return edges


    def fault_tolerance(self):
        """Compute the percentage of edges that, when removed, disconnect any of the food sources."""
        assert self.connected, 'only compute fault tolerance for fully connected graphs'

        num_connected = 0
        edges = self.edges()

        for node1, node2 in edges:
            node1_index_node2 = np.where(self.adj[node1] == node2)[0][0]
            node2_index_node1 = np.where(self.adj[node2] == node1)[0][0]

            # remove the edge
            self.adj[node1][node1_index_node2] = -1
            self.adj[node2][node2_index_node1] = -1

            # check connectedness
            if self.fullyconnected(only_foods=True):
                num_connected += 1

            # restore the edge
            self.adj[node1][node1_index_node2] = node2
            self.adj[node2][node2_index_node1] = node1

        return num_connected / len(edges)


    def mst_actual(self):
        """Compute the actual MST size over the graph just between the food sources."""
        assert self.connected, 'only compute the actual MST for fully connected graphs'
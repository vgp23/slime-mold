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
        # make sure to use a hashable coordinate type
        coord = tuple(coord)

        # use a cache dict to prevent the expensive lookup whenever possible
        try:
            if coord in self.coord_to_node_cache:
                return self.coord_to_node_cache[coord]
        except AttributeError:
            self.coord_to_node_cache = dict()

        # lookup the coordinate in the big coordinate list
        for index, c in enumerate(self.c.all_coordinates_unscaled):
            if np.array_equal(coord, c):
                self.coord_to_node_cache[coord] = index
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


    def distance_cache(self, nodes, f_distance):
        """Construct a cache that saves the distance between all node pairs."""
        dist_cache = {node1: {node2: 0 for node2 in nodes} for node1 in nodes}

        for node1 in nodes:
            for node2 in nodes:
                # skip node1 = node2 and dont compute dist(node1, node2) and
                # dist(node2, node1) twice as these are symmetric
                if node1 <= node2:
                    continue

                dist = f_distance(node1, node2)
                dist_cache[node1][node2] = dist
                dist_cache[node2][node1] = dist

        return dist_cache


    def mst(self, f_distance):
        """Compute the size of a minimum spanning tree between the food sources,
        using the given distance function which gives the distance between two
        food sources."""
        nodes = list(map(tuple, self.c.foods_unscaled))  # this is a list of coordinates
        dist_cache = self.distance_cache(nodes, f_distance)

        # keep track of which node sources we have already added to the MST
        nodes_out = list(range(1, len(nodes)))
        nodes_in = [0]
        total_size = 0

        while len(nodes_out) > 0:
            # find the minimum length edge that connects a node in node_out to node_in
            min_node = -1  # put the node index here
            min_dist = np.inf

            for node_out in nodes_out:
                for node_in in nodes_in:
                    if (d := dist_cache[nodes[node_out]][nodes[node_in]]) < min_dist:
                        min_node = node_out
                        min_dist = d

            total_size += min_dist

            nodes_out.remove(min_node)
            nodes_in.append(min_node)

        return total_size


    def f_manhatten_distance(self, coord1, coord2):
        """Compute the manhatten distance between two coordinates. Note, if both are
        in horizontal or vertical alignment and this alignment coincides with a
        wall, then we need to go around the wall, so we add 2 to the distance."""
        manhatten_distance = np.abs(coord1[0] - coord2[0]) + np.abs(coord1[1] - coord2[1])
        if ((coord1[0] == coord2[0] and coord1[0] % 2 == 1) or
            (coord1[1] == coord2[1] and coord1[1] % 2 == 1)
        ):
            manhatten_distance += 2
        return manhatten_distance


    def mst_perfect(self):
        """Compute the size of a minimum spanning tree between the food sources,
        NOT considering the actual network!"""
        return self.mst(self.f_manhatten_distance)


    def f_actual_distance(self, coord1, coord2):
        """Compute the distance between coord1 and coord2 by doing a breadth
        first search through the network."""

        node1 = self.coord_to_node(coord1)
        node2 = self.coord_to_node(coord2)

        def recurs(frontier, goal_node):
            if goal_node in frontier:
                return 0

            next_frontier = set()
            for node in frontier:
                neighbours = self.neighbours(node)

                # stop early expanding the frontier
                if goal_node in neighbours:
                    return 1

                next_frontier.update(neighbours)

            return recurs(next_frontier, goal_node) + 1

        frontier = {node1}
        return recurs(frontier, node2)


    def mst_actual(self):
        """Compute the actual MST size over the graph just between the food sources."""
        assert self.connected, 'only compute the actual MST for fully connected graphs'
        return self.mst(self.f_actual_distance)


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
        """Percentage of edges that, when removed, disconnect any of the food sources."""
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
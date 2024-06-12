import numpy as np
import copy


class Config:

    def __init__(self, seed=None):
        if seed is not None: np.random.seed(seed)

        # gui
        self.upscale = 9
        self.display_food = False
        self.display_trail = True
        self.display_agents = True
        self.display_food_sources = True
        self.display_walls = True
        self.display_history = True
        self.display_graph = True

        # walls
        self.wall_num_height = 5
        self.wall_num_width = 5

        self.wall_height = 15
        self.wall_width = 15

        self.height = self.wall_height * (2 * self.wall_num_height + 1)
        self.width = self.wall_width * (2 * self.wall_num_width + 1)

        # initialization
        self.initial_population_density = 0.5

        # agent
        self.sensor_length = 4
        self.reproduction_threshold = 15
        self.elimination_threshold = -10

        # trail
        self.trail_deposit = 5
        self.trail_damping = 0.1
        self.trail_filter_size = 3
        self.trail_weight = 0.1

        # food
        self.food_size = 3
        self.num_food = 5
        self.food_deposit = 10
        self.food_damping = 0.1
        self.food_filter_size = 3
        self.food_weight = 1 - self.trail_weight

        # starvation
        self.starvation_penalty = 0.5
        self.starvation_threshold = 0.1

        # food pickup
        self.food_pickup_threshold = 1
        self.food_pickup_limit = self.food_deposit
        self.food_drop_amount = 0.3

        # objective function
        self.triangle_agent_cutoff = 2
        self.agent_cutoff = 1
        self.patch_cutoff = 45
        self.triangle_cutoff = 30
        self.patch_edge_cutoff = 20
        self.history_size = 30

        # coordinate lists
        self.all_coordinates_unscaled, self.all_coordinates_scaled = self.coordinates()

        # generate food
        self.foods_unscaled, self.foods = self.generate_foods()


    def scale_food_coordinates(self, coordinates):
        coordinates = copy.deepcopy(coordinates)
        coordinates[:, 0] = coordinates[:, 0] * self.wall_height + self.wall_height // 2 - self.food_size // 2
        coordinates[:, 1] = coordinates[:, 1] * self.wall_width + self.wall_width // 2 - self.food_size // 2
        return coordinates


    def scale_coordinates(self, coordinates):
        coordinates = copy.deepcopy(coordinates)
        coordinates[:, 0] = coordinates[:, 0] * self.wall_height + self.wall_height // 2
        coordinates[:, 1] = coordinates[:, 1] * self.wall_width + self.wall_width // 2
        return coordinates


    def coordinates(self):
        """Return all unscaled coordinates not in walls."""
        X, Y = np.meshgrid(
            np.arange(self.wall_num_width * 2 + 1), np.arange(self.wall_num_height * 2 + 1)
        )
        coordinates = np.vstack((Y.flatten(), X.flatten())).T  # [(y, x)]
        mask = ~((coordinates[:, 0] % 2 != 0) & (coordinates[:, 1] % 2 != 0))
        coordinates = coordinates[mask]
        return coordinates, self.scale_coordinates(coordinates)


    def generate_foods(self):
        assert (
            self.food_size % 2 == 0 and self.wall_height % 2 == 0 and self.wall_width % 2 == 0
        ) or (
            self.food_size % 2 == 1 and self.wall_height % 2 == 1 and self.wall_width % 2 == 1
        ), "both food and wall needs to be odd/even"

        # sample food source coordinates
        food_choices = np.random.choice(
            range(len(self.all_coordinates_unscaled)), size=(self.num_food,), replace=False
        )
        food_coordinates = self.all_coordinates_unscaled[food_choices]

        foods_unscaled = copy.deepcopy(food_coordinates)
        foods = self.scale_food_coordinates(food_coordinates)

        return foods_unscaled, foods
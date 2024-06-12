import numpy as np
import copy


class Config:

    def __init__(self, seed=None, **kwargs):
        # gui
        self.upscale = 5
        self.display_food_sources = True
        self.display_food = False
        self.display_trail = False
        self.display_agents = False
        self.display_walls = True
        self.display_history = False
        self.display_graph = False

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
        self.num_food = 8
        self.food_span = 0.8
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
        self.history_size = 30

        # coordinate lists
        self.all_coordinates_unscaled, self.all_coordinates_scaled = self.coordinates()

        # generate food
        self.foods_unscaled, self.foods = self.generate_foods(seed)

        # set user provided settings
        for key, value in kwargs.items():
            setattr(self, key, value)

            # handle exceptions
            if key == 'food_deposit':
                self.food_pickup_limit = self.food_deposit
            if key == 'trail_weight':
                self.food_weight = 1 - self.trail_weight


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


    def generate_food_setup(self):
        """Generate a possible setup for food sources"""
        food_choices = np.random.choice(
            range(len(self.all_coordinates_unscaled)), size=(self.num_food,), replace=False
        )
        return self.all_coordinates_unscaled[food_choices]


    def food_setup_spanning(self, food_coordinates):
        """Given a setup for food sources, determine whether the minimum height
        and width span percentage are met."""
        max_x_span = 0
        max_y_span = 0

        for coord1 in food_coordinates:
            for coord2 in food_coordinates:
                y_diff, x_diff = abs(coord1 - coord2)
                max_x_span = max(max_x_span, x_diff)
                max_y_span = max(max_y_span, y_diff)

        return (
            (max_x_span >= self.food_span * (self.wall_num_width * 2 + 1)) and
            (max_y_span >= self.food_span * (self.wall_num_height * 2 + 1))
        )


    def generate_foods(self, seed):
        assert (
            self.food_size % 2 == 0 and self.wall_height % 2 == 0 and self.wall_width % 2 == 0
        ) or (
            self.food_size % 2 == 1 and self.wall_height % 2 == 1 and self.wall_width % 2 == 1
        ), "both food and wall needs to be odd/even"

        np.random.seed(seed)

        # generate food coordinates, until we got a setup that spans the
        # requirement amount of the scene
        food_coordinates = self.generate_food_setup()
        while not self.food_setup_spanning(food_coordinates):
            food_coordinates = self.generate_food_setup()

        # save scaled and unscaled coordinates
        foods_unscaled = copy.deepcopy(food_coordinates)
        foods = self.scale_food_coordinates(food_coordinates)

        return foods_unscaled, foods
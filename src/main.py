from agent import *
from scene import *
import numpy as np
import matplotlib.pyplot as plt
import collections
import pygame
import time
import copy


class Config:

    def __init__(self, seed=np.random.random()):
        self.wall_num_height = 5
        self.wall_num_width = 5

        self.wall_height = 15
        self.wall_width = 15

        self.height = self.wall_height * (2 * self.wall_num_height + 1)
        self.width = self.wall_width * (2 * self.wall_num_width + 1)

        self.upscale = 9
        self.initial_population_density = 0.5

        self.trail_deposit = 5
        self.trail_damping = 0.1
        self.trail_filter_size = 3
        self.trail_weight = 0.1

        self.chemo_deposit = 10
        self.chemo_damping = 0.1
        self.chemo_filter_size = 3
        self.chemo_weight = 1 - self.trail_weight

        self.sensor_length = 4 # DECREASED
        self.reproduction_trigger = 15
        self.elimination_trigger = -10

        # penalty for being far from food
        self.starvation_penalty = 0.5
        self.starvation_threshold = 0.1

        # food source settings
        self.food_deposit = 10
        self.food_amount = 8
        self.food_size = 3

        # food pickup
        self.food_pickup_threshold = 1
        self.food_pickup_amount = 1
        self.food_pickup_limit = self.food_deposit

        # food drop
        self.food_drop_amount = 0.3

        # visualization settings
        self.display_chemo = False
        self.display_trail = True
        self.display_agents = True
        self.display_food = True
        self.display_walls = True
        self.display_history = True
        # objective function visualization diagnostics
        self.display_graph = True

        # objective function parameters
        self.triangle_agent_cutoff = 2
        self.agent_cutoff = 1
        self.patch_cutoff = 45
        self.triangle_cutoff = 30
        self.patch_edge_cutoff = 20
        self.history_size = 30

        assert (
            self.food_size % 2 == 0 and self.wall_height % 2 == 0 and self.wall_width % 2 == 0
        ) or (
            self.food_size % 2 == 1 and self.wall_height % 2 == 1 and self.wall_width % 2 == 1
        ), "both food and wall needs to be odd/even"

        # generate random food sources

        # assume a wall and empty space to each have a width of 1,
        # sample from the empty spaces to get food locations, and
        # scale these coordinates up according to the actual dimensions.
        # np.random.seed(seed)
        X, Y = np.meshgrid(np.arange(self.wall_num_width * 2 + 1), np.arange(self.wall_num_height * 2 + 1))
        coordinates = np.vstack((Y.flatten(), X.flatten())).T  # [(y, x)]
        mask = ~((coordinates[:, 0] % 2 != 0) & (coordinates[:, 1] % 2 != 0))  # filter out wall coordinates
        coordinates = coordinates[mask]

        # sample food coordinates and scale accordingly
        food_choices = np.random.choice(range(len(coordinates)), size=(self.food_amount,), replace=False)
        food_coordinates = coordinates[food_choices]
        food_coordinates[:, 0] = food_coordinates[:, 0] * self.wall_height + self.wall_height // 2 - self.food_size // 2
        food_coordinates[:, 1] = food_coordinates[:, 1] * self.wall_width + self.wall_width // 2 - self.food_size // 2
        self.foods = food_coordinates

        # save scaled versions of the complete list of non-wall coordinates,
        # for use in objective function
        self.all_coordinates_unscaled = copy.copy(coordinates)
        coordinates[:, 0] = coordinates[:, 0] * self.wall_height + self.wall_height // 2
        coordinates[:, 1] = coordinates[:, 1] * self.wall_width + self.wall_width // 2
        self.all_coordinates_scaled = coordinates


def wait_for_spacebar():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return False


def check_keypresses(c):
    """Check if the user tries to close the program."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True
            if event.key == pygame.K_SPACE:
                return wait_for_spacebar()

            if event.key == pygame.K_h:
                c.display_history = not c.display_history
            if event.key == pygame.K_c:
                c.display_chemo = not c.display_chemo
            if event.key == pygame.K_t:
                c.display_trail = not c.display_trail
            if event.key == pygame.K_a:
                c.display_agents = not c.display_agents
            if event.key == pygame.K_f:
                c.display_food = not c.display_food
            if event.key == pygame.K_w:
                c.display_walls = not c.display_walls
            if event.key == pygame.K_g:
                c.display_graph = not c.display_graph

    return False


def scene_update(i, limit):
    while True:
        # change the scene to one before or one after
        for event in pygame.event.get():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    return None

                if event.key == pygame.K_LEFT:
                    return max(i - 1, 0)
                if event.key == pygame.K_RIGHT:
                    return min(i + 1, limit)


def draw(scene, screen, font, i=None):
    """Draw the scene to the screen."""
    # draw the scene
    surface = pygame.pixelcopy.make_surface(scene.pixelmap())
    screen.blit(surface, (0, 0))

    # draw iteration counter
    text = font.render(f'{i}' if i is not None else '', True, (88, 207, 57))
    screen.blit(text, (5, 5))

    pygame.display.update()


def visualise(scenes, c, i=None):
    """Visualise a single scene for inspection."""
    pygame.init()

    font = pygame.font.Font(None, 48)
    screen = pygame.display.set_mode(c.upscale * np.array([c.width, c.height]))
    limit = len(scenes) - 1

    if i is None:
        i = limit

    while True:
        draw(scenes[i], screen, font, i)
        i = scene_update(i, limit)

        if i is None:
            return


def run_with_gui(c, num_iter=np.inf):
    """Run a simulation with gui, this is useful for debugging and getting images."""
    pygame.init()

    font = pygame.font.Font(None, 48)
    screen = pygame.display.set_mode(c.upscale * np.array([c.width, c.height]))

    scene = Scene(c)

    i = 0
    while i < num_iter:
        scene.step()
        draw(scene, screen, font, i)

        if check_keypresses(c):
            pygame.quit()
            return scene

        i += 1

    while True:
        if check_keypresses():
            pygame.quit()
            return scene

def run_headless(c, num_iter=20000):
    """Run simulations headless on the gpu without gui."""
    scenes = [Scene(c)]

    for _ in range(num_iter):
        scenes[-1].step()
        scenes.append(copy.deepcopy(scenes[-1]))

    return scenes


def calculate_metrics(adjacency_list):
    '''Computes '''
    pass


if __name__ == '__main__':
    # generate a configuration to the experiment with
    c = Config(seed=37)
    # run an experiment with gui
    # t0 = time.time()
    scene = run_with_gui(c, num_iter=10000)
    # print(time.time() - t0)

    # run an experiment headless
    # t0 = time.time()
    # scenes = run_headless(c, num_iter=10)
    # print(time.time() - t0)
    # visualise(scenes, c)
    # print(scenes[0].trail_grid)

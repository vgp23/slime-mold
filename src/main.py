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

        self.upscale = 7
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
        self.display_chemo = True
        self.display_trail = True
        self.display_agents = True
        self.display_food = True

        assert (
            self.food_size % 2 == 0 and self.wall_height % 2 == 0 and self.wall_width % 2 == 0
        ) or (
            self.food_size % 2 == 1 and self.wall_height % 2 == 1 and self.wall_width % 2 == 1
        ), "both food and wall needs to be odd/even"

        # generate random food sources
        np.random.seed(seed)
        X, Y = np.meshgrid(np.arange(self.wall_num_width * 2 + 1), np.arange(self.wall_num_height * 2 + 1))
        coordinates = np.vstack((Y.flatten(), X.flatten())).T  # [(y, x)]
        mask = ~((coordinates[:, 0] % 2 != 0) & (coordinates[:, 1] % 2 != 0))  # filter out wall coordinates
        coordinates = coordinates[mask]
        food_choices = np.random.choice(range(len(coordinates)), size=(self.food_amount,), replace=False)
        coordinates = coordinates[food_choices]
        np.random.seed()

        # scale the food coordinates
        coordinates[:, 0] = coordinates[:, 0] * self.wall_height + self.wall_height // 2 - self.food_size // 2
        coordinates[:, 1] = coordinates[:, 1] * self.wall_width + self.wall_width // 2 - self.food_size // 2
        self.foods = coordinates


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


def check_interrupt():
    """Check if the user tries to close the program."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True

        # hacky way of pauzing the animation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                return wait_for_spacebar()

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

        if check_interrupt():
            pygame.quit()
            return

        i += 1

    while True:
        if check_interrupt():
            pygame.quit()
            return


def run_headless(c, num_iter=20000):
    """Run simulations headless on the gpu without gui."""
    scenes = [Scene(c)]

    for _ in range(num_iter):
        scenes[-1].step()
        scenes.append(copy.deepcopy(scenes[-1]))

    return scenes


if __name__ == '__main__':
    # generate a configuration to the experiment with
    c = Config(seed=37)

    # run an experiment with gui
    t0 = time.time()
    run_with_gui(c, num_iter=1000)
    print(time.time() - t0)

    # run an experiment headless
    # t0 = time.time()
    # scenes = run_headless(c, num_iter=1000)
    # print(time.time() - t0)
    # visualise(scenes, c)

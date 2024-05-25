from agent import *
from scene import *
import numpy as np
import matplotlib.pyplot as plt
import collections
import pygame
import time
import copy


class Config:

    def __init__(self, **kwargs):
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
        self.chemo_filter_size = 5
        self.chemo_weight = 1 - self.trail_weight

        self.sensor_length = 3 # DECREASED
        self.reproduction_trigger = 15
        self.elimination_trigger = -10

        # food settings
        self.food_deposit = 10
        self.food_amount = 10
        self.food_size = 3

        foods_y = np.random.choice(np.arange(self.height - self.food_size), size=(self.food_amount,))
        foods_x = np.random.choice(np.arange(self.width - self.food_size), size=(self.food_amount,))
        self.foods = np.stack((foods_y, foods_x), axis=-1)

        # visualization settings
        self.display_chemo = True
        self.display_trail = False
        self.display_agents = True
        self.display_food = True

        # # set user input configuration settings
        # for key, value in kwargs.items():
        #     setattr(self, key, value)


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
    # np.random.seed(37)
    c = Config()

    # run an experiment with gui
    t0 = time.time()
    run_with_gui(c)
    print(time.time() - t0)

    # run an experiment headless
    # t0 = time.time()
    # scenes = run_headless(c, num_iter=1000)
    # print(time.time() - t0)
    # visualise(scenes, c)

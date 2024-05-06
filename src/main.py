from agent import *
from scene import *
import numpy as np
import matplotlib.pyplot as plt
import collections
import pygame
import time


def initialize_config():
    # define all the hyperparameters for the simulation

    # original parameters
    # height, width = 200, 200
    # upscale = 4
    # initial_population_density = 0.5

    # trail_deposit = 5
    # trail_damping = 0.1
    # trail_filter_size = 3
    # trail_weight = 0.4

    # chemo_deposit = 10
    # chemo_damping = 0.2
    # chemo_filter_size = 5
    # chemo_weight = 1 - trail_weight

    # sensor_length = 7
    # reproduction_trigger = 15
    # elimination_trigger = -10

    height, width = 100, 100
    upscale = 10
    initial_population_density = 0.1

    trail_deposit = 5
    trail_damping = 0.2 # DOUBLED, less spreading
    trail_filter_size = 3
    trail_weight = 0.4

    chemo_deposit = 10
    chemo_damping = 0.2
    chemo_filter_size = 5
    chemo_weight = 1 - trail_weight

    sensor_length = 4 # DECREASED
    reproduction_trigger = 15
    elimination_trigger = -10

    food_amount = 10
    food_size = 3

    foods_y = np.random.choice(np.arange(height - food_size), size=(food_amount,))
    foods_x = np.random.choice(np.arange(width - food_size), size=(food_amount,))
    foods = np.stack((foods_y, foods_x), axis=-1)

    # visualization settings
    display_chemo = True
    display_trail = True
    display_agents = True
    display_food = True

    # pack the config info into smaller variables for convenience
    config_agent = (sensor_length, reproduction_trigger, elimination_trigger)
    config_trail = (trail_deposit, trail_damping, trail_filter_size, trail_weight)
    config_chemo = (chemo_deposit, chemo_damping, chemo_filter_size, chemo_weight)
    config_display = (display_chemo, display_trail, display_agents, display_food)
    config_food = (foods, food_amount, food_size)
    config_scene = (height, width, upscale, initial_population_density, config_display)

    return config_scene, config_food, config_agent, config_trail, config_chemo


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


def draw(scene, config_scene, screen, font, i=None):
    """Draw the scene to the screen."""
    height, width, upscale, _, config_display = config_scene

    # draw the scene
    surface = pygame.pixelcopy.make_surface(scene_pixelmap(scene, upscale, config_display))
    screen.blit(surface, (0, 0))

    # draw iteration counter
    text = font.render(f'{i}' if i is not None else '', True, (88, 207, 57))
    screen.blit(text, (5, 5))

    pygame.display.update()


def visualise(scenes, i=None):
    """Visualise a single scene for inspection."""
    config_scene, _, _, _, _ = initialize_config()
    height, width, upscale, _, _ = config_scene

    pygame.init()

    font = pygame.font.Font(None, 48)
    screen = pygame.display.set_mode(upscale * np.array([width, height]))

    if i is None:
        i = len(scenes[0]) - 1

    while True:
        draw(scenes[i], config_scene, screen, font, i)

        change_of_scene = False
        while not change_of_scene:
            # change the scene to one before or one after
            for event in pygame.event.get():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return

                    if event.key == pygame.K_LEFT:
                        i = max(i - 1, 0)
                        change_of_scene = True
                    if event.key == pygame.K_RIGHT:
                        i = min(i + 1, len(scenes[0]) - 1)
                        change_of_scene = True


def run_with_gui(config, num_iter=np.inf):
    """Run a simulation with gui, this is useful for debugging and getting images."""
    config_scene, config_food, config_agent, config_trail, config_chemo = config
    height, width, upscale, _, _ = config_scene

    pygame.init()

    font = pygame.font.Font(None, 48)
    screen = pygame.display.set_mode(upscale * np.array([width, height]))

    scene = scene_init(config_scene, config_food, config_chemo)

    i = 0
    while i < num_iter:
        scene = scene_step(scene, config_trail, config_chemo, config_agent)
        draw(scene, config_scene, screen, font, i)

        if check_interrupt():
            pygame.quit()
            return

        i += 1

    while True:
        if check_interrupt():
            pygame.quit()
            return


def run_headless(config, num_iter=20000):
    """Run simulations headless on the gpu without gui."""
    config_scene, config_food, config_agent, config_trail, config_chemo = config

    scene = scene_init(config_scene, config_food, config_chemo)
    scenes = [scene]

    for _ in range(num_iter):
        scene = scene_step(scene, config_trail, config_chemo, config_agent)

    return scenes


if __name__ == '__main__':
    # generate a configuration to the experiment with
    config = initialize_config()

    # run an experiment with gui
    t0 = time.time()
    run_with_gui(config)
    print(time.time() - t0)

    # # run an experiment headless
    # t0 = time.time()
    # scenes = run_headless(config, num_iter=1000)
    # print(time.time() - t0)

    # # visualise one scene from the headless run
    # visualise(scenes)

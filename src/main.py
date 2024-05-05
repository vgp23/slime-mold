from agent import *
from scene import *
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import matplotlib.pyplot as plt
import collections
import pygame
import time


def initialize_config(key):
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
    initial_population_density = 0.5

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

    key, subkey1, subkey2 = jr.split(key, 3)
    foods_y = jr.choice(subkey1, jnp.arange(height - food_size), shape=(food_amount,))
    foods_x = jr.choice(subkey2, jnp.arange(width - food_size), shape=(food_amount,))
    foods = jnp.stack((foods_y, foods_x), axis=-1)

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


def check_interrupt():
    """Check if the user tries to close the program."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True
    return False


def visualise(scene, key, i=None):
    """Visualise a single scene for inspection."""
    key, subkey = jr.split(key, 2)  # NOTE use the same key as for run_headless
    config_scene, _, _, _, _ = initialize_config(subkey)
    height, width, upscale, _, _ = config_scene

    pygame.init()

    font = pygame.font.Font(None, 48)
    screen = pygame.display.set_mode(upscale * jnp.array([width, height]))

    draw(scene, config_scene, screen, font, i)

    while True:
        if check_interrupt():
            break

    pygame.quit()


def draw(scene, config_scene, screen, font, i=None):
    """Draw the scene to the screen."""
    height, width, upscale, _, config_display = config_scene

    # draw the scene
    upscaled_shape = upscale * jnp.array([height, width])
    surface = pygame.pixelcopy.make_surface(scene_pixelmap(scene, upscaled_shape, config_display))
    screen.blit(surface, (0, 0))

    # draw iteration counter
    text = font.render(f'{i}' if i is not None else '', True, (88, 207, 57))
    screen.blit(text, (5, 5))

    pygame.display.update()


def run_with_gui(config, key, num_iter=jnp.inf):
    """Run a simulation with gui, this is useful for debugging and getting images."""
    config_scene, config_food, config_agent, config_trail, config_chemo = config
    height, width, upscale, _, _ = config_scene

    pygame.init()

    font = pygame.font.Font(None, 48)
    screen = pygame.display.set_mode(upscale * jnp.array([width, height]))

    key, subkey = jr.split(key, 2)
    scene = scene_init(config_scene, config_food, config_chemo, subkey)

    i = 0
    while i < num_iter:
        key, subkey = jr.split(key, 2)
        scene = scene_step(scene, config_trail, config_chemo, config_agent, subkey)

        draw(scene, config_scene, screen, font, i)

        if check_interrupt():
            pygame.quit()
            return

        i += 1

    while True:
        if check_interrupt():
            pygame.quit()
            return


def run_headless(config, key, num_iter=20000):
    """Run simulations headless on the gpu without gui."""
    config_scene, config_food, config_agent, config_trail, config_chemo = config

    key, subkey = jr.split(key, 2)
    scene = scene_init(config_scene, config_food, config_chemo, subkey)

    key, *subkeys = jr.split(key, num_iter + 1)
    _, scenes = jax.lax.scan(
        lambda scene, k: (scene_step(scene, config_trail, config_chemo, config_agent, k), scene),
        scene,
        jnp.array(subkeys),
        length=num_iter,
    )

    # the returned scenes list is of shape (len(scene), num_iter, height, width, ...)
    # i.e. scenes[0][i] contains the agent_grid at iteration i
    return scenes


if __name__ == '__main__':
    # force run all computations on the cpu
    jax.config.update('jax_platform_name', 'cpu')
    # check on which device we are running
    print(repr(jnp.square(2).addressable_data(0).devices()))

    key = jr.PRNGKey(37)

    # generate a configuration to the experiment with
    key, subkey = jr.split(key, 2)
    config = initialize_config(subkey)

    # run an experiment with gui
    t0 = time.time()
    run_with_gui(config, key)
    print(time.time() - t0)

    # # run an experiment headless
    # t0 = time.time()
    # scenes = run_headless(config, key, num_iter=100)
    # print(time.time() - t0)

    # # visualise one scene from the headless run
    # n = -1
    # scene = (scenes[0][n], scenes[1][n], scenes[2][n], scenes[3][n], scenes[4][n])
    # visualise(scene, key)

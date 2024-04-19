from agent import *
from scene import *
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import collections
import pygame


def check_interrupt():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True
    return False


def main():

    # define all the hyperparameters for the simulation
    height, width = 100, 100
    initial_population_density = 0.1
    sensor_length = 7

    trail_deposit = 5
    trail_damping = 0.1
    trail_filter_size = 3

    chemo_deposit = 10
    chemo_damping = 0.2
    chemo_filter_size = 5

    trail_weight = 0.4
    chemo_weight = 1 - trail_weight

    reproduction_trigger = 15
    elimination_trigger = -10

    key = jr.PRNGKey(37)
    key, subkey = jr.split(key, 2)

    # initialize pygame
    upscale = 15

    pygame.init()
    screen = pygame.display.set_mode(upscale * jnp.array([width, height]))
    fps_limiter = pygame.time.Clock()

    # initialize the scene data
    scene = scene_init(height, width, initial_population_density, subkey)
    print('init done')

    key, subkey = jr.split(key, 2)
    scene = scene_step(scene, subkey)
    print('step done')

    # draw the scene
    surface = pygame.pixelcopy.make_surface(scene_pixelmap(scene, upscale))
    screen.blit(surface, (0, 0))

    pygame.display.update()
    print('draw done')

    # fps_limiter.tick(10)

    while True:
        if check_interrupt():
            break

    # quit pygame
    pygame.quit()


if __name__ == '__main__':
    main()
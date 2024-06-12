from config import *
from agent import *
from scene import *
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import collections
import pygame
import time
import copy


def check_keypresses(c, pause):
    """Check if the user tries to close the program."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True, pause
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True, pause
            if event.key == pygame.K_SPACE:
                return False, not pause

            if event.key == pygame.K_h:
                c.display_history = not c.display_history
            if event.key == pygame.K_c:
                c.display_food = not c.display_food
            if event.key == pygame.K_t:
                c.display_trail = not c.display_trail
            if event.key == pygame.K_a:
                c.display_agents = not c.display_agents
            if event.key == pygame.K_f:
                c.display_food_sources = not c.display_food_sources
            if event.key == pygame.K_w:
                c.display_walls = not c.display_walls
            if event.key == pygame.K_g:
                c.display_graph = not c.display_graph

    return False, pause


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
    pause = False

    # step through the scenes
    i = 0
    while True:
        if not pause:
            i += 1
            scene.step()

        draw(scene, screen, font, i)
        stop, pause = check_keypresses(c, pause)

        if stop or i >= num_iter:
            pygame.quit()
            return scene

    # wait at the last scene
    while True:
        stop, pause = check_keypresses(c, pause)
        if stop:
            pygame.quit()
            return scene

        draw(scene, screen, font, i)


def run_headless(c, num_iter, process_id=None, results=None):
    """Run simulations headless without gui."""
    scenes = [Scene(c)]

    for _ in range(num_iter - 1):
        scenes[-1].step()
        scenes.append(copy.deepcopy(scenes[-1]))

    return scenes


def run_headless_concurrent(c, num_iter, process_id, results):
    """Run simulations headless without gui concurrently."""
    np.random.seed(process_id)
    scene = Scene(c)

    for _ in range(num_iter - 1):
        scene.step()

    if process_id is not None:
        print(f'process {process_id} finished')
        results['scenes'][process_id] = scene
        return


def run_repeated(configs, num_iter, repetitions=mp.cpu_count()):
    # use the same config for all if only one is given
    if not isinstance(configs, list):
        configs = [configs for _ in range(repetitions)]

    manager = mp.Manager()
    results = manager.dict()

    results['scenes'] = manager.dict()
    jobs = []

    for process_id, config in enumerate(configs):
        jobs.append(mp.Process(
            target=run_headless_concurrent,
            args=(config, num_iter, process_id, results)
        ))
        jobs[-1].start()

    for process in jobs:
        process.join()

    return results['scenes'].values()


def run_experiments(parameter_setups, num_food_setups, food_setup_repetitions, num_iter):
    food_seeds = list(range(num_food_setups))

    for parameter_name, parameter_values in parameter_setups.items:
        for parameter_value in parameter_values:

            configs = [
                Config(food_seed, parameter_name=parameter_value)
                for food_seed in food_seeds
            ]

            for _ in range(food_setup_repetitions):
                results = run_repeated(configs, num_iter)
                # TODO save results


if __name__ == '__main__':
    # generate a configuration to the experiment with
    # c = Config()

    # run an experiment with gui
    # t0 = time.time()
    food_seeds = list(range(mp.cpu_count() - 2))
    for seed in food_seeds:
        c = Config(seed)
        scene = run_with_gui(c)
    # print(time.time() - t0)

    # run an experiment headless
    # t0 = time.time()
    # scenes = run_headless(c, num_iter=10)
    # print(time.time() - t0)
    # visualise(scenes, c)

    parameter_setups = {
        'initial_population_density': [0.01, 0.04, 0.07, 0.1, 0.3, 0.5],
        'reproduction_threshold': [10, 15, 20, 25, 30, 35],
        'elimination_threshold': [-5, -10, -15, -20, -25, -30],
        'trail_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'starvation_penalty': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'food_drop_amount': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }
    # run_experiments(configs, mp.cpu_count() - 2, 5, 1000)


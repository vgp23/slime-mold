from agent import *
from functools import partial
import collections
import numpy as np
import scipy


# TODO make food_size static arg so it can be jit compiled
def scene_init(config_scene, config_food, config_chemo):
    height, width, _, initial_population_density, _ = config_scene
    foods, food_amount, food_size = config_food
    chemo_deposit, _, _, _ = config_chemo

    # generate the positions at which we got agents using the specified density
    mask_grid = np.random.uniform(size=(height, width)) < initial_population_density

    # generate agent data at the positions in mask_grid
    agent_grid = np.full((*mask_grid.shape, 3), 73, dtype=int)
    for y in range(len(mask_grid)):
        for x in range(len(mask_grid[y])):
            if mask_grid[y, x]:
                agent_grid[y, x] = agent_init()

    trail_grid = np.zeros((height, width))  # contains trail data

    # Contains chemo-nutrient/food data. Also need a binary mask of the initial
    # chemo grid, to keep food source values constant across iterations.
    chemo_grid = np.zeros((height, width))
    for food in foods:
        chemo_grid[food[0]:food[0] + 3, food[1]:food[1] + 3] = chemo_deposit
    food_grid = chemo_grid > 0  # food sources mask grid

    return agent_grid, mask_grid, trail_grid, chemo_grid, food_grid


def grid_coordinates(height, width):
    """Get all coordinates in a 2d grid."""
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    return np.vstack((Y.flatten(), X.flatten())).T


def out_of_bounds(pos, height, width):
    return pos[0] < 0 or pos[1] < 0 or pos[0] >= height or pos[1] >= width


def rotate_sense(agent, coordinate, scene, config_agent, config_trail, config_chemo):
    '''Rotates agent towards the sensor with the highest calculated value as per Wu et al.'''
    agent_grid, mask_grid, trail_grid, chemo_grid, _ = scene
    sensor_length, _, _   = config_agent
    _, _, _, trail_weight = config_trail
    _, _, _, chemo_weight = config_chemo

    left_sensor, right_sensor = agent_sensor_positions(agent, sensor_length)
    ls_coord = coordinate + left_sensor
    rs_coord = coordinate + right_sensor

    height, width = mask_grid.shape

    # compute values of sensors
    l_sv = 0
    if not out_of_bounds(ls_coord, height, width):
        l_sv = chemo_weight * chemo_grid[*ls_coord] + trail_weight * trail_grid[*ls_coord]

    r_sv = 0
    if not out_of_bounds(rs_coord, height, width):
        r_sv = chemo_weight * chemo_grid[*rs_coord] + trail_weight * trail_grid[*rs_coord]

    # update direction based on which is larger
    direction_idx = np.random.choice(2)
    if l_sv > r_sv:
        direction_idx = 0
    if l_sv < r_sv:
        direction_idx = 1

    new_directions = agent_sensor_directions(agent)
    agent[:2] = new_directions[direction_idx]

    return agent


def move_agent(agent, coordinate, scene, config_agent, config_trail, config_chemo, to_update):
    '''Moves an agent forward along its direction vector, by one grid position.'''
    agent_grid, mask_grid, trail_grid, chemo_grid, food_grid = scene
    trail_deposit, _, _, _ = config_trail

    new_pos = coordinate + agent[:2]
    agent[-1] = agent[-1] + 1

    # rotate agent towards sensor with highest value
    scene = agent_grid, mask_grid, trail_grid, chemo_grid, food_grid
    agent = rotate_sense(agent, new_pos, scene, config_agent, config_trail, config_chemo)

    # remove the agent from the old position
    agent_grid[*coordinate] = np.zeros(3)
    mask_grid[*coordinate] = False

    agent_grid[*new_pos] = agent
    mask_grid[*new_pos] = True
    # flag this position, so that the agent cannot be moved more than once
    # (happens when agent is moved to a position not yet iterated over)
    to_update[*new_pos] = False

    # deposit trail at new position
    trail_grid[*new_pos] = trail_grid[*new_pos] + trail_deposit


def random_orientation(agent, coordinate, scene):
    '''Randomly selects new direction for the current agent.'''
    agent_grid, mask_grid, trail_grid, chemo_grid, _ = scene

    temp_agent = agent_init()
    agent[:2] = temp_agent[:2]
    agent[-1] = agent[-1] - 1

    agent_grid[*coordinate] = agent


def reproduce(agent_grid, mask_grid, to_update, old_coordinate):
    '''Randomly initializes a new agent in the position of its parent agent, if
    the parent has exceeded the reproduction trigger threshold '''
    new_agent = agent_init()
    agent_grid[*old_coordinate] = new_agent
    mask_grid[*old_coordinate] = True
    # don't want to update the children in the current iteration
    to_update[*old_coordinate] = False


def agent_present(coordinate, scene, config_agent, config_trail, config_chemo, to_update):
    '''Computes a position update for the given agent.'''
    agent_grid, mask_grid, trail_grid, chemo_grid, food_grid = scene
    sensor_length, reproduction_trigger, elimination_trigger = config_agent

    agent = agent_grid[*coordinate]

    left_sensor, right_sensor = agent_sensor_positions(agent, sensor_length)
    ls_coord = coordinate + left_sensor
    rs_coord = coordinate + right_sensor
    height, width = mask_grid.shape

    # rotate the agent until its sensors are not out of bounds
    random_orientation(agent, coordinate, scene)

    # change agent to its rotated form, and repackage the scene
    agent_grid[*coordinate] = agent
    scene = agent_grid, mask_grid, trail_grid, chemo_grid, food_grid

    # If the elimination trigger is met, remove agent. Otherwise, attempt to move forward.
    if agent[-1] < elimination_trigger:
        agent_grid[*coordinate] = np.zeros(3)
        mask_grid[*coordinate] = False
    else:
        new_pos = coordinate + agent[:2]
        if not out_of_bounds(new_pos, height, width) and not mask_grid[*new_pos]:
            move_agent(agent, coordinate, scene, config_agent, config_trail, config_chemo, to_update)
        else:
            random_orientation(agent, coordinate, scene)

    # if the reproduction trigger is met, generate new agent in the current agent's old position
    if agent[-1] > reproduction_trigger:
        reproduce(agent_grid, mask_grid, to_update, coordinate)


def step(scene, config_agent, config_trail, config_chemo, to_update, coordinate):
    '''Performs an update to a single grid coordinate position.'''
    agent_grid, mask_grid, trail_grid, chemo_grid, food_grid = scene

    # if grid has an agent there and to_update is False in this position,
    # the agent was placed at that position during an update of a previous
    # coordinate and should be ignored.
    is_agent = mask_grid[*coordinate] & to_update[*coordinate]

    # update agent position and trail grid
    if is_agent:
        agent_present(coordinate, scene, config_agent, config_trail, config_chemo, to_update)


# TODO split so it can be jit compiled, perform kernel averaging on gpu?
def scene_step(scene, config_trail, config_chemo, config_agent):
    """Perform one update step on the scene by updating each agent in a random order."""
    agent_grid, mask_grid, trail_grid, chemo_grid, food_grid = scene
    _, trail_damping, trail_filter_size, _ = config_trail
    chemo_deposit, chemo_damping, chemo_filter_size, _ = config_chemo

    # generate a shuffled list of coordinates which determines the agent update order.
    # coordinates are only updated if an agent is on them.
    coordinates = np.random.permutation(grid_coordinates(*mask_grid.shape))

    # boolean grid, used to account for possibility that agent is moved to a
    # grid position that has not yet been iterated over, leading to an agent
    # moving multiple times.
    to_update = np.full_like(mask_grid, True)

    # step through all the coordinates and update the agents on those positions
    for coordinate in coordinates:
        step(scene, config_agent, config_trail, config_chemo, to_update, coordinate)

    # convolve both the chemo and trail with an average filter after
    # all agents have been updated + rotated.
    agent_grid, mask_grid, trail_grid, chemo_grid, food_grid = scene

    # chemo grid
    chemo_kernel = np.ones((chemo_filter_size, chemo_filter_size)) * (1 / chemo_filter_size**2)
    chemo_grid = scipy.signal.convolve2d(chemo_grid, chemo_kernel, mode='same')
    chemo_grid = chemo_grid * (1 - chemo_damping)

    # reset the values in the food sources to the default
    not_food_grid = food_grid == 0
    chemo_grid = np.multiply(not_food_grid, chemo_grid) + food_grid * chemo_deposit

    # trail grid
    trail_kernel = np.ones((trail_filter_size, trail_filter_size)) * (1 / trail_filter_size**2)
    trail_grid = scipy.signal.convolve2d(trail_grid, trail_kernel, mode='same')
    trail_grid = trail_grid * (1 - trail_damping)

    scene = agent_grid, mask_grid, trail_grid, chemo_grid, food_grid
    return scene


def scene_pixelmap(scene, upscale, config_display):
    """Create a pixelmap of the scene on the gpu that can be drawn directly."""
    agent_grid, mask_grid, trail_grid, chemo_grid, food_grid = scene
    display_chemo, display_trail, display_agents, display_food = config_display

    # create a black and white colormap based on where there are agents
    agent_colormap = ((1 - mask_grid) * 255)

    # create colormap for trails and food source, blue and red respectively
    # upscale trail and chemo maps
    trail_colormap = np.copy(trail_grid)
    chemo_colormap = np.copy(chemo_grid)
    food_colormap  = np.copy(food_grid)

    # To achieve the desired color,the target color channel is set to 255,
    # and the other two are *decreased* in proportion to the value in the
    # trail/chemo map. This makes low values close to white, and high
    # values a dark color.
    red_channel = np.full_like(agent_colormap, 255)
    green_channel = np.full_like(agent_colormap, 255)
    blue_channel = np.full_like(agent_colormap, 255)

    if display_chemo:
        # TODO make chemo spreading visual when trails are also visible
        # intensity transformation, strictly for visual purposes
        # clipping the map back to [0, 255]
        intensity = 30
        chemo_colormap = np.minimum(intensity * chemo_colormap, 255)
        chemo_colormap = np.full_like(chemo_colormap, 255) - chemo_colormap # inverting the map

        red_channel = np.full_like(chemo_colormap, 255)
        green_channel = chemo_colormap
        blue_channel = np.copy(chemo_colormap)

    if display_trail:
        # intensity transformation, strictly for visual purposes
        # clipping the map back to [0, 255]
        intensity = 10
        trail_colormap = np.minimum(intensity * trail_colormap, 255)
        trail_colormap = np.full_like(trail_colormap, 255) - trail_colormap # inverting the map

        trail_pixels = trail_colormap < 255
        not_trail_pixels = trail_colormap == 255

        red_channel = red_channel * not_trail_pixels + trail_colormap * trail_pixels
        green_channel = green_channel * not_trail_pixels + trail_colormap * trail_pixels
        blue_channel = blue_channel * not_trail_pixels + np.full_like(blue_channel, 255) * trail_pixels

    if display_agents:
        agent_pixels = agent_colormap == 0
        not_agent_pixels = agent_colormap == 255

        red_channel = red_channel * not_agent_pixels + agent_colormap * agent_pixels
        green_channel = green_channel * not_agent_pixels + agent_colormap * agent_pixels
        blue_channel = blue_channel * not_agent_pixels + agent_colormap * agent_pixels

    if display_food:
        # placing food sources on top of everything
        food_pixels = food_colormap > 0
        not_food_pixels = food_colormap == 0

        red_channel = red_channel * not_food_pixels + np.full_like(red_channel, 255) * food_pixels
        green_channel = green_channel * not_food_pixels + np.zeros_like(green_channel) * food_pixels
        blue_channel = blue_channel * not_food_pixels + np.zeros_like(blue_channel) * food_pixels

    pixelmap = np.stack((
        red_channel.astype(np.uint8),
        green_channel.astype(np.uint8),
        blue_channel.astype(np.uint8)
    ), axis=-1)

    # transpose from shape (height, width, 3) to (width, height, 3) for pygame
    transposed_pixelmap = np.transpose(pixelmap, (1, 0, 2))
    scaled_pixelmap = transposed_pixelmap.repeat(upscale, axis=0).repeat(upscale, axis=1)

    # move the data from the gpu to the cpu so pygame can draw it
    return scaled_pixelmap
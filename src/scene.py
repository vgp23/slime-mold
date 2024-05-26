from agent import *
import numpy as np
import scipy
import copy
import time
from vector import Vector2D
import pygame

class Scene:

    def __init__(self, c):
        self.c = c  # save the config

        # generate agent data at the positions in mask_grid
        # generate the positions at which we got agents using the specified density
        mask_grid = np.random.uniform(
            size=(c.height, c.width)) < c.initial_population_density

        self.agent_grid = np.full(mask_grid.shape, None, dtype=object)
        for y in range(len(mask_grid)):
            for x in range(len(mask_grid[y])):
                if mask_grid[y, x]:
                    self.agent_grid[y, x] = Agent()

        self.trail_grid = np.zeros((c.height, c.width))  # contains trail data

        # Contains chemo-nutrient/food data. Also need a binary mask of the initial
        # chemo grid, to keep food source values constant across iterations.
        self.chemo_grid = np.zeros((c.height, c.width))
        for food in c.foods:
            self.chemo_grid[
                food[0]:food[0] + c.food_size, food[1]:food[1] + c.food_size
            ] = c.food_deposit

        self.food_grid = self.chemo_grid > 0  # food sources mask grid

        # boolean grid, used to account for possibility that agent is moved to a
        # grid position that has not yet been iterated over, leading to an agent
        # moving multiple times.
        self.to_update = np.full_like(self.agent_grid, True)

        # generate the walls in the grid
        self.wall_mask = np.full_like(self.agent_grid, False)
        for y in range(c.wall_num_height):
            for x in range(c.wall_num_width):
                y_start = c.wall_height + y * 2 * c.wall_height
                x_start = c.wall_width + x * 2 * c.wall_width
                self.wall_mask[
                    y_start:y_start + c.wall_height,
                    x_start:x_start + c.wall_width
                ] = True

        # kill all agents in walls
        # self.agent_grid[self.wall_mask == True] = None


    def out_of_bounds(self, pos):
        return pos.x < 0 or pos.y < 0 or \
               pos.y >= self.c.height or pos.x >= self.c.width or \
               self.wall_mask[pos.y, pos.x]


    def rotate_sense(self, agent, coordinate):
        '''Rotates agent towards the sensor with the highest calculated value
           as per Wu et al.'''
        # lsensor, msensor, rsensor = agent.sensor_positions(agent, self.c)
        lsensor, rsensor = agent.sensor_positions(self.c)
        lcoord = coordinate + lsensor
        # mcoord = coordinate + msensor
        rcoord = coordinate + rsensor

        def sensor_value(coord):
            value = -np.inf
            if not self.out_of_bounds(coord):
                value = self.c.chemo_weight * self.chemo_grid[coord.y, coord.x] + \
                        self.c.trail_weight * self.trail_grid[coord.y, coord.x]
            return value

        # compute values of sensors
        lvalue = sensor_value(lcoord)
        # mvalue = sensor_value(mcoord)
        rvalue = sensor_value(rcoord)

        # check if the sensors are out of bounds
        # while True:
        #     if lvalue == -np.inf and rvalue == -np.inf:
        #         agent.rotate_180()
        #         agent.counter -= 1
        #         continue

        #     if lvalue == -np.inf:
        #         agent.rotate_right()
        #         agent.counter -= 1
        #         continue

        #     if rvalue == -np.inf:
        #         agent.rotate_left()
        #         agent.counter -= 1
        #         continue

        #     break

        # update direction based on which is larger
        if lvalue > rvalue:
            agent.rotate_left()
            return

        if lvalue < rvalue:
            agent.rotate_right()
            return


    def move_agent(self, agent, coordinate):
        '''Moves an agent forward along its direction vector, by one grid position.'''
        new_pos = coordinate + agent.direction()
        agent.counter += 1

        # rotate agent towards sensor with highest value
        self.rotate_sense(agent, new_pos)

        self.agent_grid[new_pos.y, new_pos.x] = copy.deepcopy(agent)
        # flag this position, so that the agent cannot be moved more than once
        # (happens when agent is moved to a position not yet iterated over)
        self.to_update[new_pos.y, new_pos.x] = False

        # remove the agent from the old position
        self.agent_grid[coordinate.y, coordinate.x] = None

        # deposit trail at the new position
        self.trail_grid[new_pos.y, new_pos.x] += self.c.trail_deposit


    def reproduce(self, coordinate):
        '''Randomly initializes a new agent in the position of its parent agent, if
        the parent has exceeded the reproduction trigger threshold.'''
        self.agent_grid[coordinate.y, coordinate.x] = Agent()
        self.to_update[coordinate.y, coordinate.x] = False


    def agent_present(self, coordinate):
        '''Computes a position update for the given agent.'''
        agent = self.agent_grid[coordinate.y, coordinate.x]

        if self.chemo_grid[coordinate.y, coordinate.x] < 0.1:
            agent.counter -= 0.4

        # If the elimination trigger is met, remove agent.
        # Otherwise, attempt to move forward.
        if agent.counter < self.c.elimination_trigger:
            self.agent_grid[coordinate.y, coordinate.x] = None
        else:
            new_pos = coordinate + agent.direction()
            if (
                not self.out_of_bounds(new_pos) and \
                self.agent_grid[new_pos.y, new_pos.x] is None
            ):
                self.move_agent(agent, coordinate)

                # if the reproduction trigger is met, generate new agent in the current agent's old position
                if self.agent_grid[new_pos.y, new_pos.x].counter > self.c.reproduction_trigger:
                    self.reproduce(coordinate)
            else:
                agent.random_direction()
                agent.counter -= 1


    def step(self):
        """Perform one update step on the scene by updating each agent in a random order."""
        # generate a shuffled list of coordinates which determines the agent update order.
        # coordinates are only updated if an agent is on them.
        X, Y = np.meshgrid(np.arange(self.c.width), np.arange(self.c.height))
        grid_coordinates = np.vstack((Y.flatten(), X.flatten())).T
        coordinates = np.random.permutation(grid_coordinates)  # [(y, x)]

        self.to_update = np.full_like(self.agent_grid, True)

        # step through all the coordinates and update the agents on those positions
        for coordinate in coordinates:
            coordinate = Vector2D(*coordinate)
            # update agent position and trail grid
            if (
                self.agent_grid[coordinate.y, coordinate.x] is not None and \
                self.to_update[coordinate.y, coordinate.x]
            ):
                self.agent_present(coordinate)

        self.diffuse()


    def diffuse(self):
        """Convolve both the chemo and trail with an average filter after all
        agents have been updated + rotated."""

        # chemo grid
        chemo_kernel = np.ones(
            (self.c.chemo_filter_size, self.c.chemo_filter_size)
        ) * (1 / self.c.chemo_filter_size**2)

        self.chemo_grid = scipy.signal.convolve2d(self.chemo_grid, chemo_kernel, mode='same')
        self.chemo_grid = self.chemo_grid * (1 - self.c.chemo_damping)

        self.chemo_grid = self.chemo_grid * (1 - self.wall_mask.astype(int))  # clip out diffusion into walls

        # reset the values in the food sources to the default
        not_food_grid = self.food_grid == 0
        self.chemo_grid = np.multiply(not_food_grid, self.chemo_grid) + \
            self.food_grid * self.c.chemo_deposit

        # trail grid
        trail_kernel = np.ones(
            (self.c.trail_filter_size, self.c.trail_filter_size)
        ) * (1 / self.c.trail_filter_size**2)

        self.trail_grid = scipy.signal.convolve2d(self.trail_grid, trail_kernel, mode='same')
        self.trail_grid = self.trail_grid * (1 - self.c.trail_damping)

        self.trail_grid = self.trail_grid * (1 - self.wall_mask.astype(int))  # clip out diffusion into walls


    def pixelmap(self):
        """Create a pixelmap of the scene on the gpu that can be drawn directly."""
        # create a black and white colormap based on
        # creating a colormap for the walls

        agent_colormap = ((1 - ((self.agent_grid != None) | self.wall_mask)) * 255)
        # agent_colormap = ((1 - self.wall_mask) * 255)

        # create colormap for trails and food source, blue and red respectively
        # upscale trail and chemo maps
        trail_colormap = np.copy(self.trail_grid)
        chemo_colormap = np.copy(self.chemo_grid)
        food_colormap  = np.copy(self.food_grid)

        # To achieve the desired color,the target color channel is set to 255,
        # and the other two are *decreased* in proportion to the value in the
        # trail/chemo map. This makes low values close to white, and high
        # values a dark color.
        red_channel = np.full_like(agent_colormap, 255)
        green_channel = np.full_like(agent_colormap, 255)
        blue_channel = np.full_like(agent_colormap, 255)

        if self.c.display_chemo:
            # TODO make chemo spreading visual when trails are also visible
            # intensity transformation, strictly for visual purposes
            # clipping the map back to [0, 255]
            intensity = 30
            chemo_colormap = np.minimum(intensity * chemo_colormap, 255)
            chemo_colormap = np.full_like(chemo_colormap, 255) - chemo_colormap # inverting the map

            red_channel = np.full_like(chemo_colormap, 255)
            green_channel = chemo_colormap
            blue_channel = np.copy(chemo_colormap)

        if self.c.display_trail:
            # intensity transformation, strictly for visual purposes
            # clipping the map back to [0, 255]
            intensity = 20
            trail_colormap = np.minimum(intensity * trail_colormap, 255)
            trail_colormap = np.full_like(trail_colormap, 255) - trail_colormap # inverting the map

            trail_pixels = trail_colormap < 255
            not_trail_pixels = trail_colormap == 255

            red_channel = red_channel * not_trail_pixels + trail_colormap * trail_pixels
            green_channel = green_channel * not_trail_pixels + trail_colormap * trail_pixels
            blue_channel = blue_channel * not_trail_pixels + np.full_like(blue_channel, 255) * trail_pixels

        if self.c.display_agents:
            agent_pixels = agent_colormap == 0
            not_agent_pixels = agent_colormap == 255

            red_channel = red_channel * not_agent_pixels + agent_colormap * agent_pixels
            green_channel = green_channel * not_agent_pixels + agent_colormap * agent_pixels
            blue_channel = blue_channel * not_agent_pixels + agent_colormap * agent_pixels

        if self.c.display_food:
            # placing food sources on top of everything
            food_pixels = food_colormap > 0
            not_food_pixels = food_colormap == 0

            red_channel = red_channel * not_food_pixels + np.full_like(red_channel, 255) * food_pixels
            green_channel = green_channel * not_food_pixels + np.zeros_like(green_channel) * food_pixels
            blue_channel = blue_channel * not_food_pixels + np.zeros_like(blue_channel) * food_pixels

        pixelmap = np.stack(
            (red_channel.astype(np.uint8), green_channel.astype(np.uint8), blue_channel.astype(np.uint8)),
            axis=-1
        )

        # transpose from shape (height, width, 3) to (width, height, 3) for pygame
        transposed_pixelmap = np.transpose(pixelmap, (1, 0, 2))
        scaled_pixelmap = transposed_pixelmap.repeat(self.c.upscale, axis=0).repeat(self.c.upscale, axis=1)

        return scaled_pixelmap
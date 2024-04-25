from agent import *
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import collections
import numpy as np


@partial(jax.jit, static_argnames=['config_scene'])
def scene_init(config_scene, key):
    height, width, _upscale, initial_population_density = config_scene

    # generate the positions at which we got agents using the specified density
    key, subkey = jr.split(key, 2)
    mask_grid = jr.uniform(subkey, (height, width)) < initial_population_density

    # generate agent data at the positions in mask_grid
    key, *subkeys = jr.split(key, height * width + 1)
    subkeys = jnp.array(subkeys).reshape((height, width, len(key)))
    agent_grid = jax.vmap(jax.vmap(
        lambda mask, key: jax.lax.cond(
            mask, lambda k: agent_init(k), lambda _k: jnp.zeros((3,)).astype(int), key
        )
    ))(mask_grid, subkeys)

    trail_grid = jnp.zeros((height, width))  # contains trail data
    chemo_grid = jnp.zeros((height, width))  # contains chemo-nutrient/food data

    return agent_grid, mask_grid, trail_grid, chemo_grid


@partial(jax.jit, static_argnames=['height', 'width'])
def _grid_coordinates(height, width):
    """Get all coordinates in a 2d grid."""
    X, Y = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    return jnp.vstack((X.flatten(), Y.flatten())).T



@jax.jit
def _step(carry, coordinate):
    ''' Performs an update to a single grid coordinate position'''

    def out_of_bounds(ls_coord, rs_coord, height, width):
        ''' Returns whether the left or right sensor(s) are out
        of bounds of the field '''

        # Goes through all possible out-of-bounds cases individually,
        # jax doesn't allow them to all be evaluated at once.
        l_oob = jnp.select(condlist=[ls_coord[0] < 0, 
                                     ls_coord[0] >= height,
                                     ls_coord[1] < 0,
                                     ls_coord[1] >= width],
                    choicelist=[True,True,True,True],
                    default=False)

        r_oob = jnp.select(condlist=[rs_coord[0] < 0, 
                                     rs_coord[0] >= height,
                                     rs_coord[1] < 0,
                                     rs_coord[1] >= width],
                    choicelist=[True,True,True,True],
                    default=False)

        return l_oob, r_oob

    def continue_rotation(carry_in):
        ''' Returns whether any of the sensors are out of bounds, or
        8 rotations have already occured, in which case agent is stuck.
        Used as the condition in the while loop within rotate_agent'''

        ls_coord, rs_coord, _, _, n_rotations, elimination_trigger,\
            height, width = carry_in
        l_oob, r_oob = out_of_bounds(ls_coord, rs_coord, height, width)
        oob = l_oob | r_oob
        not_stuck_agent = n_rotations < elimination_trigger

        return oob & not_stuck_agent

    def rotate_agent(carry_in):
        ''' Rotates agent once, according to rules in Wu et al. Part of 
        loop in agent_present() function. '''

        ls_coord, rs_coord, agent, sensor_length, n_rotations, elimination_trigger, \
            height, width = carry_in
        l_oob, r_oob = out_of_bounds(ls_coord, rs_coord, height, width)

        # use l_oob, r_oob, and current agent direction as indices to
        # determine how to rotate the agent. 
        # indexing order: (l_oob, r_oob, dx+1, dy+1), all values for
        # l_oob = 0 and r_oob = 0 should never be accessed. 
        rotation_lut = jnp.array(

                         [[[[(-5,-5),(-5,-5),(-5,-5)],
                            [(-5,-5),(-5,-5),(-5,-5)],
                            [(-5,-5),(-5,-5),(-5,-5)]],

                           [[( 0,-1),(-1,-1),(-1, 0)],
                            [( 1,-1),(-5,-5),(-1, 1)],
                            [( 1, 0),( 1, 1),( 0, 1)]]],

                          [[[(-1, 0),(-1, 1),( 0, 1)],
                            [(-1,-1),(-5,-5),( 1, 1)],
                            [( 0,-1),( 1,-1),( 1, 0)]],

                           [[( 1, 1),( 1, 0),( 1,-1)],
                            [( 0, 1),(-5,-5),( 0,-1)],
                            [(-1, 1),(-1, 0),(-1,-1)]]]])

        # convert l_oob and r_oob (booleans) to numerical indices
        l_oob_idx = jax.lax.cond(l_oob, lambda:1 , lambda:0)
        r_oob_idx = jax.lax.cond(r_oob, lambda:1 , lambda:0)

        new_direction = rotation_lut[l_oob_idx, r_oob_idx, agent[0]+1, agent[1]+1]
        # update direction and decrement movement counter
        agent = agent.at[:2].set(new_direction)
        agent = agent.at[-1].set(agent[-1]-1)

        new_left_sensor, new_right_sensor = agent_sensor_positions(agent, sensor_length)
        new_ls_coord = coordinate + new_left_sensor
        new_rs_coord = coordinate + new_right_sensor    
        # jax.debug.print('rotating: {x}', x=agent)
        n_rotations += 1

        return (new_ls_coord, new_rs_coord, agent, sensor_length, n_rotations, 
            elimination_trigger, height, width)    

    def remove_agent(agent_grid, mask_grid, coordinate, _):
        ''' Clears the grid positions at a specific coordinate. Used to
        destroy agents in the field. '''
        agent_grid = agent_grid.at[*coordinate].set(jnp.zeros(3, dtype=jnp.int32))
        mask_grid = mask_grid.at[*coordinate].set(False)

        return agent_grid, mask_grid

    def move_agent(agent_grid, mask_grid, coordinate, _):
        ''' Moves an agent forward along its velocity vector, by
        one grid position'''
        new_pos = coordinate + agent_grid[*coordinate][:2]
        agent = agent_grid[*coordinate]        
        agent = agent.at[-1].set(agent[-1]+1)
        agent_grid, mask_grid = remove_agent(agent_grid, mask_grid, coordinate, _)

        agent_grid = agent_grid.at[*new_pos].set(agent)
        mask_grid = mask_grid.at[*new_pos].set(True)

        return agent_grid, mask_grid

    def new_orientation(agent_grid, mask_grid, coordinate, key):
        ''' Randomly selects new velocity for the current agent '''
        agent = agent_grid[*coordinate]           
        temp_agent = agent_init(key)
        agent = agent.at[:2].set(temp_agent[:2])
        agent = agent.at[-1].set(agent[-1]-1)

        agent_grid = agent_grid.at[*coordinate].set(agent)

        return agent_grid, mask_grid


    def attempt_move(agent_grid, mask_grid, coordinate, key):
        ''' Attempts to move an agent one square in the direction of 
        their velocity vector. If the square is occupied, randomly
        chooses a new velocity vector and assigns it to the agent.'''
        new_pos = coordinate + agent_grid[*coordinate][:2]
        agent_grid, mask_grid = jax.lax.cond(
            mask_grid[*new_pos]==False, 
            move_agent,
            new_orientation,
            agent_grid, mask_grid, coordinate, key)

        return agent_grid, mask_grid


    def agent_present(agent, sensor_length, agent_grid, mask_grid, coordinate, 
        reproduction_trigger, elimination_trigger, key, height, width):
        ''' Computes a position update for the given agent'''
        # compute coordinates of each sensor
        left_sensor, right_sensor = agent_sensor_positions(agent, sensor_length)
        ls_coord = coordinate + left_sensor
        rs_coord = coordinate + right_sensor

        # rotate the agent until its sensors are not out of bounds
        carry_in = (ls_coord, rs_coord, agent, sensor_length, 0, elimination_trigger,
            height, width)

        carry_out = jax.lax.while_loop(continue_rotation, rotate_agent, carry_in)
        _, _, rotated_agent, _, _, _, _, _ = carry_out

        # if the elimination trigger is met, remove agent. Otherwise, attempt
        # to move forward.
        agent_grid, mask_grid = jax.lax.cond(
            rotated_agent[-1]<= elimination_trigger,
            remove_agent, attempt_move, 
            agent_grid, mask_grid, coordinate, key)

        return agent_grid, mask_grid

    def agent_absent(agent, sensor_length, agent_grid, mask_grid, coordinate, 
        reproduction_trigger, elimination_trigger, key, height, width):
        ''' Returns an empty grid position'''
        # jax.debug.print("No agent")
        return agent_grid, mask_grid

    agent_grid, mask_grid, trail_grid, chemo_grid, \
    sensor_length, reproduction_trigger, elimination_trigger, key = carry
    key, subkey = jr.split(key, 2)

    # jax.debug.print('Coordinate: {x}', x=coordinate)
    agent = agent_grid[*coordinate]
    # jax.debug.print('Original agent: {x}', x=agent)    
    is_agent = mask_grid[*coordinate]
    height, width = mask_grid.shape

    carry_in = (agent, sensor_length, agent_grid, mask_grid, coordinate,
        reproduction_trigger, elimination_trigger, key, height, width)

    agent_grid, mask_grid = jax.lax.cond(
        is_agent, agent_present, agent_absent, *carry_in)

    # jax.debug.print('Rotated agent: {x}', x=agent_grid[*coordinate])   

    carry_out = (agent_grid, mask_grid, trail_grid, chemo_grid, 
        sensor_length, reproduction_trigger, elimination_trigger, key)

    return carry_out, None


@partial(jax.jit, static_argnames=['config_trail', 'config_chemo', 'config_agent'])
def scene_step(scene, config_trail, config_chemo, config_agent, key):

    """Perform one update step on the scene by updating each agent in a random order."""
    agent_grid, mask_grid, trail_grid, chemo_grid = scene
    sensor_length, reproduction_trigger, elimination_trigger = config_agent
    # generate a shuffled list of coordinates which determines the agent update order.
    # coordinates are only updated if an agent is on it. 
    coordinates = jr.permutation(key, _grid_coordinates(*mask_grid.shape))
    results, _ = jax.lax.scan(_step, (*scene, *config_agent, key), coordinates)

    agent_grid, mask_grid, trail_grid, chemo_grid, _, _, _, _ = results

    return agent_grid, mask_grid, trail_grid, chemo_grid


# @partial(jax.jit, static_argnames=['upscaled_shape'])
def scene_pixelmap(scene, upscaled_shape):
    """Create a pixelmap of the scene on the gpu that can be drawn directly."""
    agent_grid, mask_grid, trail_grid, chemo_grid = scene

    # TODO add chemo grid to pixelmap
    # TODO add trail grid to pixelmap

    # create a black and white colormap based on where there are agents
    colormap = ((1 - mask_grid) * 255).astype(jnp.uint8)
    # upscale the black and white colormap
    colormap = jax.image.resize(colormap, upscaled_shape, method='nearest')

    # create three color channels based on the mask grid
    pixelmap = jnp.stack((colormap, colormap, colormap), axis=-1)

    # transpose from shape (height, width, 3) to (width, height, 3) for pygame
    transposed_pixelmap = jnp.transpose(pixelmap, (1, 0, 2))

    # move the data from the gpu to the cpu so pygame can draw it
    return jax.device_put(transposed_pixelmap, device=jax.devices('cpu')[0])

# if __name__ == '__main__':
#     height = 100
#     width= 150
#     upscale = 5 

#     sensor_length = 2
#     sensor_length = 7
#     reproduction_trigger = 15
#     elimination_trigger = -10


#     initial_population_density = 0.99
#     config_scene = (height, width, upscale, initial_population_density)
#     key = jr.PRNGKey(42)

#     agent_grid, mask_grid, trail_grid, chemo_grid = scene_init(config_scene, key)

#     scene = (agent_grid, mask_grid, trail_grid, chemo_grid)
#     config_agent = (sensor_length, reproduction_trigger, elimination_trigger)

#     print(agent_grid.shape)

#     agent_grid, mask_grid, trail_grid, chemo_grid = scene_step(scene, 0,0, config_agent, key)
#     # jax.debug.print("done")

#     print(_grid_coordinates(height, width)[-50:])





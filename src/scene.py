from agent import *
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import collections


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


def _step(carry, coordinate):
    old_agent_grid, old_mask_grid, trail_grid, chemo_grid, new_mask_grid, new_agent_grid = carry

    agent = old_agent_grid[coordinate]
    # left_sensor, right_sensor = agent_sensor_positions(agent, 5) # TODO sensor_length)

    new_mask_grid = new_mask_grid.at[coordinate].set(old_mask_grid[coordinate])     # TODO: remove me
    new_agent_grid = new_agent_grid.at[coordinate].set(old_agent_grid[coordinate])  # TODO: remove me

    carry = (old_agent_grid, old_mask_grid, trail_grid, chemo_grid, new_mask_grid, new_agent_grid)
    return carry, None


@partial(jax.jit, static_argnames=['config_trail', 'config_chemo', 'config_agent'])
def scene_step(scene, config_trail, config_chemo, config_agent, key):
    """Perform one update step on the scene by updating each agent in a random order."""
    old_agent_grid, old_mask_grid, trail_grid, chemo_grid = scene

    new_mask_grid = jnp.full_like(old_mask_grid, False)
    new_agent_grid = jnp.zeros_like(old_agent_grid)

    # generate a shuffled list of coordinates which determines the agent update order
    coordinates = jr.permutation(key, _grid_coordinates(*old_mask_grid.shape))

    # update each coordinate
    results, _ = jax.lax.scan(_step, (*scene, new_mask_grid, new_agent_grid), coordinates)
    _, _, trail_grid, chemo_grid, new_mask_grid, new_agent_grid = results

    return new_agent_grid, new_mask_grid, trail_grid, chemo_grid


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
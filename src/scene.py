from agent import *
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import collections


@partial(jax.jit, static_argnames=['height', 'width', 'initial_population_density'])
def scene_init(height, width, initial_population_density, key):
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

    new_mask_grid = new_mask_grid.at[coordinate].set(old_mask_grid[coordinate])     # TODO: remove me
    new_agent_grid = new_agent_grid.at[coordinate].set(old_agent_grid[coordinate])  # TODO: remove me

    carry = (old_agent_grid, old_mask_grid, trail_grid, chemo_grid, new_mask_grid, new_agent_grid)
    return carry, None


@jax.jit
def scene_step(scene, key):
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
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import collections


World = collections.namedtuple('World', ['agent_grid', 'trail_grid', 'chemo_grid'])


def world_init(height, width, initial_population_density, key):
    agent_grid = jnp.zeros((height, width))
    trail_grid = jnp.zeros((height, width))
    chemo_grid = jnp.zeros((height, width))

    return World(agent_grid, trail_grid, chemo_grid)


def main():
    height, width = 3, 5
    initial_population_density = 0.5
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

    world = world_init(height, width, initial_population_density, subkey)
    print(world)

    key, *subkeys = jr.split(key, 10)
    for subkey in subkeys:
        agent = agent_init(subkey)
        print(agent.dir.x, agent.dir.y)


if __name__ == '__main__':
    main()
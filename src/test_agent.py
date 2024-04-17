from .agent import *
import unittest

class TestAgent(unittest.TestCase):

    def test_sensor_positions(self):
        sensor_length = 1
        agent_sensors = [
            #  agent     left      right
            #  x   y     x   y     x   y
            [(-1, -1), (-1,  0), ( 0, -1)],
            [( 0, -1), (-1, -1), ( 1, -1)],
            [( 1, -1), ( 0, -1), ( 1,  0)],

            [( 1,  0), ( 1, -1), ( 1,  1)],

            [( 1,  1), ( 1,  0), ( 0,  1)],
            [( 0,  1), ( 1,  1), (-1,  1)],
            [(-1,  1), ( 0,  1), (-1,  0)],

            [(-1,  0), (-1,  1), (-1, -1)],
        ]

        for agent_dir, *sensors in agent_sensors:
            agent = Agent(Vec2(*agent_dir), 0)
            left, right = agent_sensor_positions(agent, sensor_length)

            self.assertTrue(
                jnp.array_equal(sensors[0], left),
                f'{agent_dir}: {sensors[0]} != {left}'
            )
            self.assertTrue(
                jnp.array_equal(sensors[1], right),
                f'{agent_dir}: {sensors[0]} != {right}'
            )
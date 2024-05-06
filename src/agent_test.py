from agent import *
import unittest


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent_sensors = [
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


    def test_sensor_positions_far(self):
        sensor_length = 5
        for agent_dir, *sensors in self.agent_sensors:
            agent = np.array([*agent_dir, 0])
            left, right = agent_sensor_positions(agent, sensor_length)

            sensors = sensor_length * np.array(sensors)

            self.assertTrue(
                np.array_equal(sensors[0], left),
                f'{agent_dir}: {sensors[0]} != {left}'
            )
            self.assertTrue(
                np.array_equal(sensors[1], right),
                f'{agent_dir}: {sensors[1]} != {right}'
            )


    def test_sensor_positions(self):
        sensor_length = 1
        for agent_dir, *sensors in self.agent_sensors:
            agent = np.array([*agent_dir, 0])
            left, right = agent_sensor_positions(agent, sensor_length)

            self.assertTrue(
                np.array_equal(sensors[0], left),
                f'{agent_dir}: {sensors[0]} != {left}'
            )
            self.assertTrue(
                np.array_equal(sensors[1], right),
                f'{agent_dir}: {sensors[1]} != {right}'
            )
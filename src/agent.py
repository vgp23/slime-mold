from vector import Vector2D
import numpy as np


class Agent:

    # [(y, x)]
    directions = np.array([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)])

    def __init__(self):
        """Generate a random new agent."""
        self.index = np.random.randint(len(Agent.directions))
        self.counter = 0


    def random_direction(self):
        self.index = np.random.randint(len(Agent.directions))


    def direction(self):
        return Vector2D(*Agent.directions[self.index])


    def sensor_directions(self):
        """Compute the directions of the sensors using a sensor map."""
        lsensor = Vector2D(*Agent.directions[(self.index - 1) % len(Agent.directions)])
        # msensor = Vector2D(*Agent.directions[self.index])
        rsensor = Vector2D(*Agent.directions[(self.index + 1) % len(Agent.directions)])
        return np.array([lsensor, rsensor])


    def sensor_positions(self, c):
        """Compute the relative positions of the sensors using a sensor map."""
        return self.sensor_directions() * c.sensor_length


    def rotate_left(self):
        self.index = (self.index - 1) % len(Agent.directions)


    def rotate_right(self):
        self.index = (self.index + 1) % len(Agent.directions)


    def rotate_180(self):
        self.index = (self.index + 4) % len(Agent.directions)
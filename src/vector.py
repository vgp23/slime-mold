class Vector2D():

    def __init__(self, y, x):
        self.x = x
        self.y = y


    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector2D(y,x)


    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Vector2D(y,x)


    def __mul__(self, scalar):
        assert not isinstance(scalar, Vector2D), 'Scalar only you dumbass'
        x = self.x * scalar
        y = self.y * scalar
        return Vector2D(y,x)


    def __div__(self, scalar):
        assert not isinstance(scalar, Vector2D), 'Scalar only you dumbass'
        x = self.x / scalar
        y = self.y / scalar
        return Vector2D(y,x)


if __name__ == "__main__":
    vector = Vector2D(3,5)
    print(isinstance(vector, Vector2D))
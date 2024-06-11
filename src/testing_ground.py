import numpy as np
import matplotlib.pyplot as plt

def get_triangle_masks(n):

    center = (n // 2, n // 2)
    # generate all possible indices
    indices = np.indices((n, n)).reshape(2, -1).T

    # generate one mask (the top)
    top = indices[(indices[:, 0] - indices[:, 1] <= 0) & (indices[:, 0] + indices[:, 1] <= n - 1)]
    top_mask = np.zeros((n,n))
    top_mask[top[:,0], top[:,1]] = 1

    # rotate it 90 degrees to get the rest

    return top_mask, \
        np.rot90(top_mask, k=3), \
        np.rot90(top_mask, k=2), \
        np.rot90(top_mask, k=1)

if __name__ == '__main__':

    # Example usage
    n = 7  # Size of the square
    top, right, down, left = get_triangle_masks(n)

    plt.figure()
    plt.imshow(left)
    plt.show()
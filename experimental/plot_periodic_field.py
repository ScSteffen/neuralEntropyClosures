import matplotlib.pyplot as plt
import numpy as np


def main():
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 20, 60)

    z = np.cos(y[:, np.newaxis]) * np.sin(x)

    fig, ax = plt.subplots()

    # filled contours
    im = ax.contourf(x, y, z, 100)

    # contour lines
    im2 = ax.contour(x, y, z, colors='k')

    fig.colorbar(im, ax=ax)
    
    return 0


if __name__ == '__main__':
    main()

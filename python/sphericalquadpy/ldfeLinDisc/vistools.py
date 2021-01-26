import matplotlib.pyplot as plt


def scatterplot(points, weights, square=1):
    x = points[:, 0]
    z = points[:, 2]
    area = 1e7 * weights**2
    plt.close("all")

    cm = plt.cm.get_cmap('Dark2')
    plt.subplot(1, 2, 1)
    plt.scatter(x, z, s=area, c=square, alpha=0.99, cmap=cm)
    plt.title("quadrature points for 1/3rd octant\ncircle area = weights")
    n_bins = 20
    # We can set the number of bins with the `bins` kwarg
    plt.subplot(1, 2, 2)
    plt.hist(weights / sum(weights) * len(weights), bins=n_bins)
    plt.xticks(rotation=90)
    plt.xlabel("ratio to mean weight")
    plt.ylabel("frequency of given weight")
    plt.title("weight distribution")
    plt.show()

'''
Script to call different plots and illustrative methods
'''

'''
Script to
Author: Steffen Schotthoefer
Version: 0.0
Date 22.10.2021
'''

from optparse import OptionParser
from src.utils import load_density_function, plot_1d
import matplotlib.pyplot as plt


def main():
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-d", "--degree", dest="degree", default=0,
                      help="max degree of moment", metavar="DEGREE")
    parser.add_option("-m", "--model", dest="model", default=11,
                      help="choice of network model", metavar="MODEL")
    parser.add_option("-p", "--processingmode", dest="processingmode", default=1,
                      help="gpu mode (1). cpu mode (0) ", metavar="PROCESSINGMODE")
    parser.add_option("-s", "--spatialDimension", dest="spatialDimension", default=3,
                      help="spatial dimension of closure", metavar="SPATIALDIM")

    (options, args) = parser.parse_args()
    options.degree = int(options.degree)
    options.spatial_dimension = int(options.spatialDimension)
    options.model = int(options.model)
    options.processingmode = int(options.processingmode)

    # --- End Option Parsing ---

    [x, _, kinetic_f] = load_density_function("test_a10_ev5.csv")

    for i in range(kinetic_f.shape[0]):
        plt.plot(x[0], kinetic_f[i, :])

    plt.ylim(0, 3)
    plt.savefig("test_a10_ev5")
    # for i in range(int(kinetic_f.shape[0] / 5)):
    #    kinetic_list = [kinetic_f[i + 0], kinetic_f[i + 1], kinetic_f[i + 2], kinetic_f[i + 3], kinetic_f[i + 4]]
    #    plot_1d(x, kinetic_list, show_fig=False, log=False, name='kinetics_kond3_' + str(i).zfill(3), ylim=[0, 3],
    #            xlim=[x[0, 0], x[0, -1]])

    return True


if __name__ == '__main__':
    main()

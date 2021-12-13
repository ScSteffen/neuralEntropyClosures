'''
Script to
Author: Steffen Schotthoefer
Version: 0.0
Date 22.10.2021
'''

from src.solver import MNSolver2D, MNSolver1D
from optparse import OptionParser
import tensorflow as tf
import os


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

    # witch to CPU mode, if wished
    if options.processingmode == 0:
        # Set CPU as available physical device
        # Set CPU as available physical device
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if tf.test.gpu_device_name():
            print('GPU found. Using GPU')
        else:
            print("Disabled GPU. Using CPU")

    if options.spatial_dimension == 1:
        solver = MNSolver1D.MNSolver1D(traditional=False, polyDegree=options.degree, model_mk=options.model)
        solver.solve(max_iter=20000, t_end=0.1)
    if options.spatial_dimension == 2:
        solver = MNSolver2D.MNSolver2D(traditional=False, model_mk=options.model)
        solver.solve(maxIter=2000, t_end=1)

    return True


if __name__ == '__main__':
    main()

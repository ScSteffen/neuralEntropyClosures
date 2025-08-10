''' 
brief: Scripts for postprocessing
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def postprocessData():
    dataframeList = []
    ax = plt.subplot()

    for i in range(1, 41):
        name = "history_" + str(i).zfill(3) + "_.csv"

        df = pd.read_csv(name)
        dataframeList.append(df)

    epochs = dataframeList[0]["epoch"].to_numpy()
    data = np.zeros((40, epochs.size))

    for i in range(40):
        data[i, :] = dataframeList[i]["loss"].to_numpy()

    for i in range(epochs.size):
        meanLoss = 0

    ax.plot(epochs, data[i, :])
    ax.set_yscale('log')
    ax.set_ylim([1e-2, 10])
    plt.show()
    return 0


if __name__ == '__main__':
    postprocessData()

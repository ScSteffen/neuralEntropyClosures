'''
small script to plot cpp output
'''

import matplotlib.pyplot as plt
import csv
import numpy as np
plt.style.use("kitish")


# load csv file
hList = list()
uList = list()
alphaList = list()

# --- Load moments u ---
uList = list()
alphaList = list()
hList = list()
hdualList = list()
f = open("cppData_primal.csv", 'r')
with f:
    reader = csv.reader(f)

    for row in reader:
        uList.append(float(row[1]))
        alphaList.append(float(row[2]))
        hList.append(float(row[3]))
f.close()

u = np.asarray(uList)
alpha = np.asarray(alphaList)
h = np.asarray(hList)

f = open("cppData_dual.csv", 'r')
with f:
    reader = csv.reader(f)

    for row in reader:
        uList.append(float(row[1]))
        alphaList.append(float(row[2]))
        hdualList.append(float(row[3]))
f.close()

h_dual = np.asarray(hdualList)

#plot entries
uplt = u[0:1000]
alphaplt = alpha[0:1000]
hplt = h[0:1000]
hDualplt = h_dual[0:1000]
print(hplt)
print(hDualplt)

#x = np.linspace(0,10,100)
#y = x*x;
#plt.plot(x,y)
# plot stuff
#plt.plot(uplt, alphaplt)
plt.plot(uplt, hplt)
#plt.plot(uplt, hDualplt)
plt.ylabel('h(u)')
plt.xlabel('u')
#plt.legend(['alpha', 'h', 'h dual'])
plt.yscale('linear')
plt.xscale('linear')
plt.show()

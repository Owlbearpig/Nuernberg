# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:13:56 2020

@author: JanO
"""

from THz.importing import import_tds_gui, extract_rob_data, selectloc, selectFile, \
    import_tds  # , extract_timestamp, import_klimalogger
import numpy as np
import re
import matplotlib.pyplot as plt
from THz.preprocessing import butter_highpass_filter, butter_lowpass_filter, offset, fft
from functions import search_folder
from constants import data_dir_day4_nail
import os.path

datafiles = search_folder(dir_location=data_dir_day4_nail, fileextension='.npz')
datafiles = [file for file in datafiles if 'BigDelamination' not in file]

dx, dy = 1, 2

data = []
for i, file in enumerate(datafiles):
    t, a, names, path, _ = import_tds(file)
    t = t[0, :]
    X = []
    Y = []
    for name in names:
        test = re.split(r" mm", name)
        x = test[-3]
        y = test[-2]
        x = re.split(r"-", x)
        if x[-2] == "":
            x = - np.round(float(x[-1]), 0)
        else:
            x = np.round(float(x[-1]), 0)
        y = re.split(r"-", y)
        y = np.round(float(y[-1]))
        X.append(x)
        Y.append(y)
    X = np.asanyarray(X)
    Y = np.asanyarray(Y)
    if i == 0:
        X += 0
        Y += 267

    # SETTING THE FULL ImAGE SIZE
    if i == 0:
        xmin = np.min(np.min(X))
        xmax = np.max(np.max(X))
        ymin = np.min(np.min(Y))
        ymax = np.max(np.max(Y))
    else:
        temp = np.min(np.min(X))
        if temp < xmin:
            xmin = temp
        temp = np.max(np.max(X))
        if temp > xmax:
            xmax = temp
        temp = np.min(np.min(Y))
        if temp < ymin:
            ymin = temp
        temp = np.max(np.max(Y))
        if temp > ymax:
            ymax = temp

    t, a = offset(t, a, 3)
    a = butter_highpass_filter(a, 0.25, fs=1 / 0.1)

    f, A = fft(t, a)
    data.append([t, a, names, path, X, Y, f, A])

xunique = np.arange(xmin, xmax + dx, dx)
yunique = np.arange(ymin, ymax + dx, dy)
amp = np.zeros((len(yunique), len(xunique), 3001))
AMP = np.zeros((len(yunique), len(xunique), 3001))
f = f[0, :]
idf = (f > 0.5) & (f < 0.8)
I = np.zeros((len(yunique), len(xunique)))

for dat in data:
    count = 0
    xscan = dat[4]
    yscan = dat[5]
    a = dat[1]
    A = dat[7]
    for x1, y1, a1, A1 in zip(xscan, yscan, a, A):
        idx = xunique == x1
        idy = yunique == y1
        if (sum(idx) == 1) & (sum(idy) == 1):
            amp[idy, idx, :] = a1
            AMP[idy, idx, :] = np.abs(A1)
            I[idy, idx] = np.trapz(np.abs(A1[idf]) ** 2)
            count += 1
    print(count)
#np.save('IntensityData_0.5-0.8THz', I)
#np.save('IntensityImageAxes_0.5-0.8THz', np.array([xunique, yunique]))

print(np.sum(I, axis=0))
plt.plot(np.sum(I, axis=0))
plt.show()

plt.figure()
plt.pcolor(xunique, yunique, I, cmap='jet')
clbr1 = plt.colorbar()
clbr1.ax.set_ylabel('Intensity (arb. units)')
plt.axis('equal')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
#plt.savefig("Intensity_0.5-0.8THz.png", dpi=300)
plt.show()

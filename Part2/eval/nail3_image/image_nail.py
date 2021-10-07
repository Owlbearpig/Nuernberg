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
import pickle

datafiles = search_folder(dir_location=data_dir_day4_nail, fileextension='.npz')
datafiles = [file for file in datafiles if 'BigDelamination' not in file]


def parse_datafiles():
    image_data_points = []
    for i, file in enumerate(datafiles):
        t, a, names, path, _ = import_tds(file)
        t = t[0, :]
        X, Y = [], []
        for name in names:
            test = re.split(r" mm", name)
            x, y = test[-3], test[-2]
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
            xmin, xmax = np.min(np.min(X)), np.max(np.max(X))
            ymin, ymax = np.min(np.min(Y)), np.max(np.max(Y))
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
        image_data_points.append([t, a, names, path, X, Y, f, A])

    return image_data_points, [xmin, xmax, ymin, ymax]


image_data_file = 'temp_image_data_dump'

try:
    image_data = pickle.load(open(image_data_file, 'rb'))
except FileNotFoundError:
    image_data = parse_datafiles()
    pickle.dump(image_data, open(image_data_file, 'wb'))

image_data_points, [xmin, xmax, ymin, ymax] = image_data
dx, dy = 1, 2
f_min, f_max = 0.5, 0.8
t_min_idx, t_max_idx = 0, 3001

xunique = np.arange(xmin, xmax + dx, dx)
yunique = np.arange(ymin, ymax + dx, dy)
amp = np.zeros((len(yunique), len(xunique), t_max_idx))
AMP = np.zeros((len(yunique), len(xunique), t_max_idx))
f = image_data_points[-1][6][0, :]
idf = (f > f_min) & (f < f_max)
I = np.zeros((len(yunique), len(xunique)))
t = image_data_points[-1][0]

for data_point in image_data_points:
    count = 0
    xscan = data_point[4]
    yscan = data_point[5]
    a = data_point[1]
    A = data_point[7]
    for x1, y1, a1, A1 in zip(xscan, yscan, a, A):
        idx = xunique == x1
        idy = yunique == y1
        if (sum(idx) == 1) & (sum(idy) == 1):
            amp[idy, idx, :] = a1
            AMP[idy, idx, :] = np.abs(A1)
            I[idy, idx] = np.trapz(np.abs(A1[idf]) ** 2)
            count += 1
    print(count)


def f_space_I_column_sum():
    plt.plot(xunique, np.sum(I, axis=0))
    plt.xlabel('x (mm)')
    plt.ylabel(f'Summed I, {f_min}-{f_max} THz, (arb. units)')
    plt.title('Sum along axis 0 (Sum together each column)')
    plt.show()


def t_space_amp_column_sum(t0_idx = 500):
    plt.plot(xunique, np.sum(np.sum(np.abs(amp[:, :, t0_idx:]), axis=2), axis=0))
    plt.xlabel('x (mm)')
    plt.ylabel(f'Summed amplitude, {t[t0_idx]}-{t[-1]} ps, (arb. units)')
    plt.title('Sum together each column after main pulse')
    plt.show()


def plot_amp_image(t0_idx = 500):
    plt.figure()
    plt.pcolor(xunique, yunique, np.sum(np.abs(amp[:, :, t0_idx:]), axis=2), cmap='jet')
    clbr1 = plt.colorbar()
    clbr1.ax.set_ylabel('Amplitude (arb. units)')
    plt.axis('equal')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.show()


def plot_AMP_image():
    plt.figure()
    plt.pcolor(xunique, yunique, I, cmap='jet')
    clbr1 = plt.colorbar()
    clbr1.ax.set_ylabel('Intensity (arb. units)')
    plt.axis('equal')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    #plt.savefig("Intensity_0.5-0.8THz.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    # Nail search ...
    f_space_I_column_sum()
    t_space_amp_column_sum()
    plot_amp_image()
    plot_AMP_image()

import matplotlib.pyplot as plt
from constants import data_dir_part1
from pathlib import Path
import numpy as np
from THz.preprocessing import offset, fft, butter_highpass_filter


def highpass_filter(t, y):
    cutoff, order = 0.2, 9
    fs = 1.0 / (t[1] - t[0])

    y = butter_highpass_filter(y, cutoff, fs, order)

    return t, y

ref_file = Path(r'Day1_X0-150_Y0-270/Reference/2020-08-25T12-35-30.514602-Reference_20avg-X-25.000 mmY-20.000 mm.txt')

ref_path = data_dir_part1 / ref_file

ref = np.loadtxt(ref_path)

t, y = ref[:, 0], ref[:, 1]

plt.plot(t, y)
plt.title('original')
plt.show()

t, y = highpass_filter(t, y)

plt.plot(t, y)
plt.title('filtered')
plt.show()

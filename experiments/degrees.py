import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


prefix = sys.argv[1]
num = int(sys.argv[2])

orig_degrees_filename = "orig_degrees.out"
degrees_filename = "degrees.out"

orig_degrees = []
degrees = []

orig_n = 0
n = 0

with open(prefix + orig_degrees_filename, "r") as f:
    for line in f:
        v = int(line.split()[0])
        d = int(line.split()[1])
        orig_degrees.append(d)
        orig_n = v+1
f.close()

with open(prefix + degrees_filename, "r") as f:
    for line in f:
        v = int(line.split()[0])
        d = int(line.split()[1])
        degrees.append(d)
        n = v+1
f.close()

# plt.plot(range(0,orig_n,orig_n//num), orig_degrees, label="original")
# plt.plot(range(0,n,n//num), degrees, label="preprocessed")

plt.scatter(range(0,orig_n,orig_n//num), orig_degrees, s=(mpl.rcParams['lines.markersize']), label="original")
plt.scatter(range(0,n,n//num), degrees, s=(mpl.rcParams['lines.markersize']), label="preprocessed")

plt.yscale("log")
plt.ylim(None, 10**5)

plt.xlabel("Vertex")
plt.ylabel("Degree")

plt.legend()

plt.show()

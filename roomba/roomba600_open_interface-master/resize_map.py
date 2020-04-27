import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

def main():
    map = np.loadtxt("map.txt", dtype=int)
    cmap = colors.ListedColormap(['white','gray','black'])
    plt.figure(figsize=(6.4,4.8), dpi=100)
    plt.pcolor(map,cmap=cmap,edgecolors='k', linewidths=0.5)
    fname = "resized_map.png"
    plt.savefig(fname)

if __name__ == "__main__":
    main()
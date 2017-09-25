import sys
import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pygame as pg
import time
from pydub import AudioSegment

color_index = {}

def load_colors(labels):
    # Load colors, then assign colors to labels
    colors = {}
    with open("colors.txt", "r") as file:
        for line in file:
            colors[line.split()[0]] = line.split()[1]
    for label in labels:
        prefix = label[:2] # Colors are assigned based on the first two symbols of label
        try:
            color_index[label] = colors[prefix]
        except:
            color_index[label] = "black" # If label not assigned a color return blue
    return color_index

def plot(labels, x_coords, y_coords, coords, sound_path, color_index):
    """
    Plot and play corresponding sound when hovering over datapoint.
    """

    # Initialize audio engine
    sample_rate = AudioSegment.from_mp3(sound_path + os.listdir(sound_path)[0]).frame_rate
    pg.mixer.init(frequency=sample_rate)
    pg.init()
    
    def onhover(event):
        # Play sound when hover
        x = event.xdata
        y = event.ydata
        sound_label = labels[calc_nearest_point((x, y), coords)]
        sound_file = sound_path + '/' + str(sound_label) + '.wav'
        pg.mixer.Sound(sound_file).play()
        print("Playing: ", sound_label)
        time.sleep(0.01) # Small pause to avoid crashing

    # Create plot and add datapoints
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label, x, y in zip(labels, x_coords, y_coords):
        ax.scatter(x, y, color=color_index[label], label=label[:2], s=100, alpha=0.4) # plot
        #ax.annotate(label[:2], xy=(x, y), xytext=(0, 0), textcoords='offset points') # Uncomment for textlabels on plot
    if sound_path:
        fig.canvas.mpl_connect('motion_notify_event', onhover)
    ax.grid(True)
    plt.show()

def calc_nearest_point(node, nodes):
        nodes = np.asarray(nodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2)

if __name__ == '__main__':
    labels, x_coords, y_coords, coords = pickle.load(open(sys.argv[1], "rb" ))
    sound_path = "sounds/"
    color_index = load_colors(labels)
    plot(labels, x_coords, y_coords, coords, sound_path, color_index)


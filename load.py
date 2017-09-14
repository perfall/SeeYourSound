import sys
import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pygame as pg

def plot(labels, x_coords, y_coords, coords, sound_path, color_index):
    pg.mixer.init(frequency=16000)
    pg.init()
    #pg.mixer.pre_init(8000) 

    def onclick(event):
        x = event.xdata
        y = event.ydata
        sound_label = labels[calc_nearest_point((x, y), coords)]
        sound_file = sound_path + '/' + str(sound_label) + '.wav'
        pg.mixer.Sound(sound_file).play()
        #play_command = 'afplay ' + sound_path + '/' + str(sound_label) + '.wav'
        print(sound_label)
        #os.system(play_command)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label, x, y in zip(labels, x_coords, y_coords):
        ax.scatter(x, y, color=color_index[label]) # plot
        ax.annotate(label[:2], xy=(x, y), xytext=(0, 0), textcoords='offset points')
    if sound_path:
        fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def calc_nearest_point(node, nodes):
        nodes = np.asarray(nodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2)

if __name__ == '__main__':
    file = open(sys.argv[1], "rb")
    labels, x_coords, y_coords, coords, sound_path, color_index = pickle.load(file)
    file.close()
    plot(labels, x_coords, y_coords, coords, sound_path, color_index)


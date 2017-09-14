 # -*- coding: utf-8 -*-
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
from som import SOM # File som.py should be in folder
import unicodedata
import time
import pickle


class Cluster:

    def __init__(self):
        self.colors = "aqua beige black blue brown chartreuse coral crimson cyan gold green indigo lavender lime magenta olive orange pink red tan yellow".split()
        self.color_index = {}
        self.sound_path = False
        self.new_file =  "saved_plots/" + "-".join([str(el) for el in time.localtime()[0:5]]) + ".txt"

    def load_data(self, spec_path, sound_path, subset):
        """
        Set path to sounds.
        Load spectrograms from dir and convert to array.
        Load labels.
        """
        self.sound_path = sound_path
        X_vectors = []
        X_labels = []
        for file in os.listdir(spec_path):
            file = unicodedata.normalize('NFC', file)
            if file.startswith('.'):# or file.startswith('A') or file.startswith('O'):
                continue
            if random.randint(1, 100) > subset:
                continue
            current_spectrogram = Image.open(spec_path+"/"+file).convert('L')
            X_vectors.append(np.array(current_spectrogram.getdata()))
            X_labels.append(file[:-4])

        # Load colors
        colors = {}
        with open("colors.txt", "r") as file:
            for line in file:
                print(line.split())
                colors[line.split()[0]] = line.split()[1]


        # Assign colors to vowels, specifically adjusted for one's data
        
        for label in X_labels:
            v = label.split("_")[0]
            try:
                self.color_index[label] = colors[v]
            except:
                print("Can't find assigned colors")
                self.color_index[label] = colors["blue"]
        return np.array(X_vectors), X_labels
        
    def grid_size(self, n):
        """
        Calculate optimal grid-size for SOM, as close to square form as possible.
        """
        root = int(abs(n**0.5))
        if root**2 == n:
            return root, root
        return root, root+1

    def train_som(self, spec_path, sound_path, subset):
        """
        Train the SOM on the vectors and get each vectors bmu (best matching unit).
        """
        
        X_vectors, X_labels = self.load_data(spec_path, sound_path, subset)
        rows, columns = self.grid_size(len(X_labels)) # Calculate grid-size
        
        som = SOM(rows, columns, len(X_vectors[0]), 200, alpha=0.3)
        som.train(X_vectors)
        
        coords = som.map_vects(X_vectors) # get coordinates
        x_coords = [c1 for c1, c2 in coords]
        y_coords = [c2 for c1, c2 in coords]
        self.plot(X_labels, x_coords, y_coords, coords)
        
    def train_tsne(self, spec_path, sound_path, subset):
            X_vectors, X_labels = self.load_data(spec_path, sound_path, subset)
            tsne = TSNE(n_components=2, perplexity=15, random_state=0)
            Y = tsne.fit_transform(X_vectors)
            self.plot(X_labels, Y[:, 0], Y[:, 1], Y)

    def plot(self, labels, x_coords, y_coords, coords):
        def onclick(event):
            x = event.xdata
            y = event.ydata
            sound_label = labels[self.calc_nearest_point((x, y), coords)]
            play_command = 'afplay ' + self.sound_path + '/' + str(sound_label) + '.wav'
            print(sound_label)
            os.system(play_command)

        file = open(self.new_file, "wb")
        pickle.dump((labels, x_coords, y_coords, coords, self.sound_path, self.color_index), file)
        file.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, x, y in zip(labels, x_coords, y_coords):
            ax.scatter(x, y, color=self.color_index[label]) # plot
            ax.annotate(label[:2], xy=(x, y), xytext=(0, 0), textcoords='offset points')
        if self.sound_path:
            fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def calc_nearest_point(self, node, nodes):
        nodes = np.asarray(nodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2)

if __name__ == '__main__':
    cluster = Cluster()
    if len(sys.argv) >= 4:
        sound_path = sys.argv[3]
    else:
        sound_path = False
    if len(sys.argv) == 5:
        subset = int(sys.argv[4])
    else:
        subset = 100
    if sys.argv[1] == "som":
        cluster.train_som(sys.argv[2], sound_path, subset)
    elif sys.argv[1] == "tsne":
        cluster.train_tsne(sys.argv[2], sound_path, subset)
    else:
        print("Incorrect argument, try 'som' or 'tsne'.")
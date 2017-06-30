from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
from sklearn.manifold import TSNE
import random

from som import SOM # File som.py should be in folder

class Cluster:

    def __init__(self):
        self.colors = "aqua beige black blue brown chartreuse coral crimson cyan gold green indigo lavender lime magenta olive orange pink red tan yellow".split()
        self.color_index = {}

    def load_data(self, spec_path):
        """
        Load spectrograms from dir and convert to array.
        Load labels.
        """
        X_vectors = []
        X_labels = []
        for file in os.listdir(spec_path):
            current_spectrogram = Image.open(spec_path+"/"+file).convert('L')
            X_vectors.append(np.array(current_spectrogram.getdata()))
            X_labels.append(file[:5][-2:])
        for label in set(X_labels):
            if len(self.colors) == 0:
                print("WARNING: Too many vowels, colors will not be assigned.")
                for l in set(X_labels):
                    self.color_index[l] = "blue"
                break
            self.color_index[label] = self.colors.pop(random.randint(0, len(self.colors)-1)) # Assign a color to each vowel
        return np.array(X_vectors), X_labels
        
    def grid_size(self, n):
        """
        Calculate optimal grid-size, as close to square form as possible.
        """
        root = int(abs(n**0.5))
        if root**2 == n:
            return root, root
        return root, root+1

    def train_som(self, spec_path):
        """
        Train the SOM on the vectors and get each vectors bmu (best matching unit).
        """
        
        X_vectors, X_labels = self.load_data(spec_path)

        rows, columns = self.grid_size(len(X_labels)) # Calculate grid-size
        
        som = SOM(rows, columns, len(X_vectors[0]), 200, alpha=0.3)
        som.train(X_vectors)
        
        mapped = som.map_vects(X_vectors) # get coordinates
        
        x_coors, y_coors = [], [] # reformat coordinates
        for c1, c2 in mapped:
            x_coors.append(c1)
            y_coors.append(c2)
        
        for label, x, y in zip(X_labels, x_coors, y_coors):
            plt.scatter(x, y, color=self.color_index[label]) # plot
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.show()

    def train_tsne(self, spec_path):
        X_vectors, X_labels = self.load_data(spec_path)

        tsne = TSNE(n_components=2, perplexity=15, random_state=0)
    
        Y = tsne.fit_transform(X_vectors)
        for label, x, y in zip(X_labels, Y[:, 0], Y[:, 1]):
            plt.scatter(x, y, color=self.color_index[label])
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.show()

if __name__ == '__main__':
    cluster = Cluster()
    if sys.argv[1] == "som":
        cluster.train_som(sys.argv[2])
    elif sys.argv[1] == "tsne":
        cluster.train_tsne(sys.argv[2])
    else:
        print("Incorrect argument, try 'som' or 'tsne'.")
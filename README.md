# Clustering Spectrograms with Self-Organizing Maps and or t-SNE

The goal of this project is an architecture that takes raw spoken data as input, extracts vowel candidates, and clusters them onto a 2D map. For now, the focus will be on the latter part, specifically the clustering of vowel sounds (as spectrograms) in a self-organizing map (SOM). The project will be updated, for now however the program receieves spectrograms as input, performs clustering, and plots the filenames.

## Libraries
Several packages are needed:

* TensorFlow (SOM)
* scikit-learn (t-SNE)
* matplotlib
* numpy
* PIL

## Usage
The main script takes two arguments, clustering algorithm('som' or 'tsne') and path_to_folder with spectrograms. SOM is extremly slow for now, start with a low amount of images and try it out. 

A test set of spectrograms are provided, simply run:
```
python3 main.py som testdata/spectrograms
```

or

```
python3 main.py tsne testdata/spectrograms
```


## Acknowledgments

Credit to Sachin Joglekar for the [som.py script](https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/).


# SeeYourSound: Explore your audio in an interactive map

This is the first stage of an architecture that receives audio as input, extracts segments of the audio, and positions these segments onto a 2D (for now) interactive map based on how they sound. This is done by generating spectrograms from segments of the source audio and then use these as input into a self-organizing map.

## Usage

Put the audio you want to use in a directory (1 or several files), if color-coding is of interest make sure that the first two letters of the audio name corresponds to the assignment in 'colors.txt'. For instance, if you want the segments of the file that starts with "MA" to be blue and the ones that starts with "WA" to be red, the following lines should be in 'colors.txt':

```
MA blue
WA red
```

Use the script 'audio2spec.py' to process your audio and generate segmented audio and spectrograms. Provide audio-dir and output-dir as arguments:

```
python3 audio2spec.py /path/to/audio /path/to/output
```

If you want to manually set the segment size include a textfile as a third argument labeled with filename-start_ms-end_ms-label, e.g.:

```
1.wav 799 809 AL
1.wav 888 898 IS 
2.wav 1005 1015 EL 
2.wav 1169 1179 AS 
```

Then use the script 'spec2map.py' to start building the map. Two algorithms are available, SOM and t-SNE, although SOM is better at preserving the topological aspects of the high dimensional data, t-SNE is much faster and great for testing and playing around. Also provide the path to the output you just created:

```
python3 spec2map.py tsne /path/to/output
python3 spec2map.py som /path/to/output
```

If you only want to use a subset of your audio, an optional third argument can be added. If you want to use 25% of the audio:

```
python3 spec2map.py som /path/to/output 25
```

After training a plot is generated with data points corresponding to the sound segments, hover over the datapoints to play the audio.

## Save and Load

When building a map a pickled file along with 'colors.txt' and 'load.py' will be added to the output dir. You can then run the 'load.py' script from this folder (and adjust colors) to view the map again without training.

```
python3 load.py pickled_file
```

## Settings

The settings file can be adjusted. 'y' refers to resolution in y-axis, 65 is default. 'X' is pixels per second, default 1000. 'ms_step' refers to size of segments in milliseconds, default 100ms.

## Test Data

A test sample is included with speech audio:

```
python3 audio2spec.py test/audio test/output
python3 spec2map.py tsne test/output

```

It might take a minute to load.

## Acknowledgments

Credit to Sachin Joglekar for the [som.py script](https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/).


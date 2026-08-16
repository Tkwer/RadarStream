# MIT License
#
# Copyright (c) 2020 Sebastian Höfer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
This is an example how to use an ImageView with Matplotlib Colormaps (cmap).
The function 'cmapToColormap' converts the Matplotlib format to the internal
format of PyQtGraph that is used in the GradientEditorItem. The function
itself has no dependencies on Matplotlib! Hence the weird if clauses with
'hasattr' instead of 'isinstance'.
"""


from collections.abc import Callable, Sequence

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph


def matplotlib_colormap_names():
    """Return colormap names using Matplotlib's public API across versions."""
    try:
        import matplotlib
    except ImportError:
        # The built-in custom map remains available when Matplotlib is optional.
        return []

    registry = getattr(matplotlib, "colormaps", None)
    if registry is not None:
        return sorted(registry)

    # ``pyplot.colormaps`` predates the registry introduced in Matplotlib 3.5.
    from matplotlib import pyplot as plt

    return sorted(plt.colormaps())


def get_matplotlib_colormap(name):
    """Resolve a colormap without relying on the removed ``cm.cmap_d`` API."""
    import matplotlib

    registry = getattr(matplotlib, "colormaps", None)
    if registry is not None:
        return registry[name]

    from matplotlib import cm

    return cm.get_cmap(name)


def cmapToColormap(cmap, nTicks=64):
    """
    Converts a Matplotlib cmap to pyqtgraphs colormaps. No dependency on matplotlib.
    Parameters:
    *cmap*: Cmap object. Imported from matplotlib.cm.*
    *nTicks*: Number of ticks to create when dict of functions is used. Otherwise unused.
    """

    # Case #1: a dictionary with 'red'/'green'/'blue' values as list of ranges (e.g. 'jet')
    # The parameter 'cmap' is a 'matplotlib.colors.LinearSegmentedColormap' instance ...
    if hasattr(cmap, '_segmentdata'):
        colordata = getattr(cmap, '_segmentdata')
        if ('red' in colordata) and isinstance(colordata['red'], Sequence):

            # collect the color ranges from all channels into one dict to get unique indices
            posDict = {}
            for idx, channel in enumerate(('red', 'green', 'blue')):
                for colorRange in colordata[channel]:
                    posDict.setdefault(colorRange[0], [-1, -1, -1])[idx] = colorRange[2]

            indexList = list(posDict.keys())
            indexList.sort()
            # interpolate missing values (== -1)
            for channel in range(3):  # R,G,B
                startIdx = indexList[0]
                emptyIdx = []
                for curIdx in indexList:
                    if posDict[curIdx][channel] == -1:
                        emptyIdx.append(curIdx)
                    elif curIdx != indexList[0]:
                        for eIdx in emptyIdx:
                            rPos = (eIdx - startIdx) / (curIdx - startIdx)
                            vStart = posDict[startIdx][channel]
                            vRange = (posDict[curIdx][channel] - posDict[startIdx][channel])
                            posDict[eIdx][channel] = rPos * vRange + vStart
                        startIdx = curIdx
                        del emptyIdx[:]
            for channel in range(3):  # R,G,B
                for curIdx in indexList:
                    posDict[curIdx][channel] *= 255

            rgb_list = [[i, posDict[i]] for i in indexList]

        # Case #2: a dictionary with 'red'/'green'/'blue' values as functions (e.g. 'gnuplot')
        elif ('red' in colordata) and isinstance(colordata['red'], Callable):
            indices = np.linspace(0., 1., nTicks)
            luts = [np.clip(np.array(colordata[rgb](indices), dtype=float), 0, 1) * 255 \
                    for rgb in ('red', 'green', 'blue')]
            rgb_list = zip(indices, list(zip(*luts)))

    # If the parameter 'cmap' is a 'matplotlib.colors.ListedColormap' instance, with the attributes 'colors' and 'N'
    elif hasattr(cmap, 'colors') and hasattr(cmap, 'N'):
        colordata = getattr(cmap, 'colors')
        # Case #3: a list with RGB values (e.g. 'seismic')
        if len(colordata[0]) == 3:
            indices = np.linspace(0., 1., len(colordata))
            scaledRgbTuples = [(rgbTuple[0] * 255, rgbTuple[1] * 255, rgbTuple[2] * 255) for rgbTuple in colordata]
            rgb_list = zip(indices, scaledRgbTuples)

        # Case #4: a list of tuples with positions and RGB-values (e.g. 'terrain')
        # -> this section is probably not needed anymore!?
        elif len(colordata[0]) == 2:
            rgb_list = [(idx, (vals[0] * 255, vals[1] * 255, vals[2] * 255)) for idx, vals in colordata]

    # Case #X: unknown format or datatype was the wrong object type
    else:
        raise ValueError("[cmapToColormap] Unknown cmap format or not a cmap!")

    # Convert the RGB float values to RGBA integer values
    return list([(pos, (int(r), int(g), int(b), 255)) for pos, (r, g, b) in rgb_list])


def pg_get_cmap(cmap):
    if(cmap=='customize'):
        position = np.arange(64)
        position = position / 64
        position[0] = 0
        position = np.flip(position)
        colors = [[62, 38, 168, 255], [63, 42, 180, 255], [65, 46, 191, 255], [67, 50, 202, 255], [69, 55, 213, 255],
                [70, 60, 222, 255], [71, 65, 229, 255], [70, 71, 233, 255], [70, 77, 236, 255], [69, 82, 240, 255],
                [68, 88, 243, 255],
                [68, 94, 247, 255], [67, 99, 250, 255], [66, 105, 254, 255], [62, 111, 254, 255], [56, 117, 254, 255],
                [50, 123, 252, 255],
                [47, 129, 250, 255], [46, 135, 246, 255], [45, 140, 243, 255], [43, 146, 238, 255], [39, 150, 235, 255],
                [37, 155, 232, 255],
                [35, 160, 229, 255], [31, 164, 225, 255], [28, 129, 222, 255], [24, 173, 219, 255], [17, 177, 214, 255],
                [7, 181, 208, 255],
                [1, 184, 202, 255], [2, 186, 195, 255], [11, 189, 188, 255], [24, 191, 182, 255], [36, 193, 174, 255],
                [44, 195, 167, 255],
                [49, 198, 159, 255], [55, 200, 151, 255], [63, 202, 142, 255], [74, 203, 132, 255], [88, 202, 121, 255],
                [102, 202, 111, 255],
                [116, 201, 100, 255], [130, 200, 89, 255], [144, 200, 78, 255], [157, 199, 68, 255], [171, 199, 57, 255],
                [185, 196, 49, 255],
                [197, 194, 42, 255], [209, 191, 39, 255], [220, 189, 41, 255], [230, 187, 45, 255], [239, 186, 53, 255],
                [248, 186, 61, 255],
                [254, 189, 60, 255], [252, 196, 57, 255], [251, 202, 53, 255], [249, 208, 50, 255], [248, 214, 46, 255],
                [246, 220, 43, 255],
                [245, 227, 39, 255], [246, 233, 35, 255], [246, 239, 31, 255], [247, 245, 27, 255], [249, 251, 20, 255]]
        colors = np.flip(colors, axis=0)
        color_map = pyqtgraph.ColorMap(position, colors)
        return color_map
    else:

        pos, rgba_colors = zip(*cmapToColormap(cmap))
        # Set the colormap
        pgColormap =  pyqtgraph.ColorMap(pos, rgba_colors)
        return pgColormap



# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    import matplotlib.cm

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtGui.QApplication([])

        # Create window with ImageView widget
        win = pyqtgraph.GraphicsLayoutWidget()
        win.resize(800,800)
        win.show()

        imv = pyqtgraph.ImageItem()

        view = win.addViewBox()
        view.addItem(imv)
        win.setCentralWidget(view)
        win.show()
        win.setWindowTitle('Example: Matplotlib colormaps in pyqtgraph ImageView')

        # Create random data set with noisy signals
        data = np.random.normal(size=(200, 200))
        gradient = np.linspace(0., 10., 200)
        data += gradient[:, np.newaxis]


        # Convert a matplotlib colormap into a list of (tickmark, (r,g,b)) tuples
        pos, rgba_colors = zip(*cmapToColormap(matplotlib.cm.coolwarm))
        # Set the colormap
        pgColormap =  pyqtgraph.ColorMap(pos, rgba_colors)
        imv.setLookupTable(pgColormap.getLookupTable())

        # Display the data
        imv.setImage(data)

        QtGui.QApplication.instance().exec_()

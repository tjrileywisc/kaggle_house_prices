
import numpy as np
import pandas as pd
import pickle
from preprocess_data import preprocess
import subprocess
import os
import seaborn as sns
import matplotlib as mpl
from scipy.stats import skew
import matplotlib.pyplot as plt

from bokeh.plotting import figure, show, output_file
from bokeh.models import LabelSet, ColumnDataSource, HoverTool, PanTool, BoxZoomTool, ResetTool

def plot_outlier(x,y):
    tmp = x.dropna()
    skew_value = skew(tmp)
    y = np.log1p(y)
    print('sample length: %s and skew: %s'%(len(x), skew_value))
    fig,axs = plt.subplots(1,2,figsize=(8,3))
    sns.boxplot(x, orient = 'v', ax = axs[0])
    sns.regplot(x, y, ax = axs[1])
    plt.tight_layout()

if not os.path.exists("./data/train_df.pickle"):
    preprocess()


# check for outliers
# from https://www.kaggle.com/life2short/house-prices-advanced-regression-techniques/eda-and-outliers/discussion



# plot t-sne and color with binned home prices
# if they don't seem to cluster well, then we need to work on features
train_df = pd.read_pickle("./data/train_df.pickle")
train_features = train_df.values
train_targets = pickle.load(open("./data/train_targets.pickle", "rb"))

binned = np.histogram(train_targets)

# flatten to 2d with largevis
np.savetxt("./data/feature_vectors.txt", train_features, header = "%s %s" % (len(train_features), train_features[0].shape[0]), comments = "")

p = subprocess.call("largevis.exe -input %s -output %s" % ("./data/feature_vectors.txt", "./data/largevis.data"))

low_dim_embs = np.loadtxt("./data/largevis.data", skiprows = 1)

# produce an html file with the embeddings plotted in 2d space
x = low_dim_embs[:, 0]
y = low_dim_embs[:, 1]
radii = np.ones_like(x)*0.1
colors = [
    "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)
]
source = ColumnDataSource(data=dict(x=x,  y=y, labels=binned))

## when you hover over words, show them in a tooltip
hover = HoverTool(
        tooltips=[
            ("", "@labels"),
        ]
    )


TOOLS="crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select"

p = figure(tools=[hover, PanTool(), BoxZoomTool(), ResetTool()],  plot_width = 1500, plot_height = 900)

p.scatter(x = 'x', y = 'y', radius=radii, fill_color=colors, fill_alpha=0.6, source = source, line_color=None)
p.axis.visible = None

output_file("word_embeddings.html", title="word embeddings")

## do some cleanup to remove some files that are huge
os.remove("./data/feature_vectors.txt")
os.remove("annoy_index_file")
os.remove("knn_graph.txt")

## open a browser and show the file
show(p)  
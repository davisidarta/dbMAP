"""
Utility functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

def embedding_plot(X, y, digits, title, show_images = True):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # X is the embedding, y are the targets
    plt.figure(figsize= [800,800], dpi = 600)
    ax = plt.subplot()
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=5, c=y)
    
    if show_images == True:
        shown_images = np.array([[1., 1.]])
        for i in range(X.shape[0]):
            if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < 3e-3: continue
            shown_images = np.r_[shown_images, [X[i]]]
            ax.add_artist(offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]))
     
    plt.xticks([]), plt.yticks([])
    plt.title(title, fontsize = 20)
                        

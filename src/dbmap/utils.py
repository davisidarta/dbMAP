"""
Utility functions
"""

import matplotlib.pyplot as plt


def dbmap_plot(res, title=None, fontsize=18, labels=None, pt_size=None, marker='o', opacity=1):
    plt.scatter(
        res[:, 0],
        res[:, 1],
    s=pt_size,
    c=labels,
    marker=marker,
    alpha=opacity
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(title, fontsize=fontsize)
                        

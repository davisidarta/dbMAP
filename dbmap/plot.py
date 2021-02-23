import sys
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("The package 'umap-learn' is required. Please install it 'with pip3 install umap-learn'.")
    sys.exit()

def scatter_plot(res, title=None, fontsize=18, labels=None, pt_size=None, marker='o', opacity=1):
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

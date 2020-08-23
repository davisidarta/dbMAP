import numpy as np
import pandas as pd
from kneed import KneeLocator

def multiscale(res, n_eigs=None, sensitivity=10, verbose=True):
    """Determine multi scale space of the data
          :param n_eigs: Number of eigen vectors to use. If None specified, the number
                 of eigen vectors will be determined using eigen gap identification.
          :return: Multi scaled data matrix
    """
    if n_eigs is None:
        vals = np.ravel(res["EigenValues"])
        n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
        if n_eigs < 3:
            n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 3

        if n_eigs < 30:
            if verbose:
                print('Selecting number of components based on potential curve.')
            # Try selecting n_eigs based on the second derivative
            vals = np.array(res['EigenValues'])
            k_use = range(len(vals))
            kn = KneeLocator(list(k_use), vals, S=sensitivity, curve='convex', direction='decreasing')
            n_eigs = kn.knee

    # Multiscale diffusion - account for all t's
    use_eigs = list(range(1, n_eigs))
    eig_vals = np.ravel(res["EigenValues"][use_eigs])
    data = res["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))

    data = pd.DataFrame(data, index=res["EigenVectors"].index)
    if verbose:
        if n_eigs is not None:
            print('Selected and multiscaled ' + str(round(n_eigs)) +
                  ' diffusion components.')
        else:
            print('Automatically selected and multiscaled ' + str(round(n_eigs)) +
                  ' diffusion components.')
    return data

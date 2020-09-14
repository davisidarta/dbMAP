import numpy as np
import pandas as pd

def multiscale(res, n_eigs=None):
    """Determine multi scale space of the data
          :param n_eigs: Number of eigen vectors to use. If None specified, the number
                 of eigen vectors will be determined using eigen gap identification.
          :return: Multi scaled data matrix
    """
    if n_eigs is None:
        vals = np.array(res["EigenValues"])
        n_eigs = np.sum( vals > 0, axis=0)
    # Multiscale diffusion
    use_eigs = list(range(n_eigs))
    ev = res['EigenValues']
    eig_vals = np.ravel(ev.reindex(use_eigs))
    ms_data = res["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))
    ms_data = pd.DataFrame(ms_data, index=res["EigenVectors"].index)
    print('Automatically selected and multiscaled ' + str(round(n_eigs)) +
          ' diffusion components.')
    return ms_data

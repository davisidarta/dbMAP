import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class multiscale(TransformerMixin, BaseEstimator):

    def __init__(self,
                 n_eigs=None
                 ):
        """Determine multi scale space of the data
        :param n_eigs: Number of eigen vectors to use. If None specified, the number
               of eigen vectors will be determined using eigen gap identification.
        :return: Multi scaled data matrix
        """
        self.n_eigs = n_eigs
       # self.sensitivity = sensitivity
       # self.plot = plot

    def fit(self, res):
        # if self.n_eigs != None:
          #  if self.plot == True:
           #     ev = res['EigenValues']            # TODO: implement an optional knee selector
            #    x = range(1, len(ev)+1)
             #   y = ev
              #  kneedle = KneeLocator(x, y, S=self.sensitivity, curve='convex', direction='decreasing')
               # kneedle.plot_knee()

        if self.n_eigs is None:
            vals = np.ravel(res["EigenValues"])
            self.n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
            if self.n_eigs < 3:
                self.n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 3
            if self.n_eigs < 3:
                self.n_eigs = self.n_eigs + 3
            print('Automatically selected and multiscaled ' + str(round(self.n_eigs)) +
                  ' diffusion components.')


        return self

    def transform(self, res):

        # Scales the data
        use_eigs = list(range(0, self.n_eigs))
        ev = res['EigenValues']
        eig_vals = np.ravel(ev.reindex(use_eigs))
        data = res["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))
        data = pd.DataFrame(data, index=res["EigenVectors"].index)

        return data

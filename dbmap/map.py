import time
import umap
from sklearn.base import TransformerMixin

class Mapper(TransformerMixin):
    """
    Layouts diffusion structure with UMAP to achieve dbMAP dimensional reduction. This class refers to the lower
    dimensional representation of diffusion components obtained through an adaptive diffusion maps algorithm initially
    proposed by [Setty18]. Alternatively, other diffusion approaches can be used, such as
    # To do: Fazer a adaptacao p outros algoritmos de diff maps
    :param n_components: int (optional, default 2). The dimension of the space to embed into. This defaults to 2 to
    provide easy visualization, but can reasonably be set to any integer value in the range 2 to K, K being the number
    of samples or diffusion components to embedd.
    :param n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points) used for
    manifold approximation. Larger values result in more global views of the manifold, while smaller values result in
    more local data being preserved. In general values should be in the range 2 to 100.
    :param n_jobs: Number of threads to use in calculations. Defaults to all but one.
    :param min_dist: The effective minimum distance between embedded points. Smaller values will result in a more
    clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will
    result on a more even dispersal of points. The value should be set relative to the spread value, which determines
    the scale at which embedded points will be spread out.
    :param spread: The effective scale of embedded points. In combination with min_dist this determines how
    clustered/clumped the embedded points are.
    :param learning_rate: The initial learning rate for the embedding optimization.
    :return: dbMAP embeddings.
    """
    def __init__(self,
                 n_components=2,
                 n_neighbors=15,
                 metric='euclidean',
                 output_metric='euclidean',
                 n_epochs=None,
                 learning_rate=1.5,
                 init='spectral',
                 min_dist=0.6,
                 spread=1.5,
                 low_memory=False,
                 set_op_mix_ratio=1.0,
                 local_connectivity=1.0,
                 repulsion_strength=1.0,
                 negative_sample_rate=5,
                 transform_queue_size=4.0,
                 a=None,
                 b=None,
                 random_state=None,
                 angular_rp_forest=False,
                 target_n_neighbors=-1,
                 target_metric='categorical',
                 target_weight=0.5,
                 transform_seed=42,
                 force_approximation_algorithm=False,
                 verbose=False,
                 unique=False):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.output_metric = output_metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.min_dist = min_dist
        self.spread = spread
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.transform_queue_size = transform_queue_size
        self.a = a
        self.b = b
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.unique = unique

    def fit(self, data, y=0):
        # 'y' defines a null variable for compatibility purposes only.

        simple_umap = umap.UMAP(n_components=self.n_components,
                                metric=self.metric,
                                output_metric=self.output_metric,
                                n_epochs=self.n_epochs,
                                learning_rate=self.learning_rate,
                                init=self.init,
                                min_dist=self.min_dist,
                                spread=self.spread,
                                low_memory=self.low_memory,
                                set_op_mix_ratio=self.set_op_mix_ratio,
                                local_connectivity=self.local_connectivity,
                                repulsion_strength=self.repulsion_strength,
                                negative_sample_rate=self.negative_sample_rate,
                                transform_queue_size=self.transform_queue_size,
                                a=self.a,
                                b=self.b,
                                random_state=self.random_state,
                                angular_rp_forest=self.angular_rp_forest,
                                target_n_neighbors=self.target_n_neighbors,
                                target_metric=self.target_metric,
                                target_weight=self.target_weight,
                                transform_seed=self.transform_seed,
                                force_approximation_algorithm=self.force_approximation_algorithm,
                                verbose=self.verbose,
                                unique=self.unique).fit_transform(data)
        return simple_umap

    def fit_transform(self, data, y=0):
        # 'y' defines a null variable for compatibility purposes only.
        start_time = time.time()
        N = data.shape[0]

        simple_umap = umap.UMAP(n_components=self.n_components,
                                metric=self.metric,
                                output_metric=self.output_metric,
                                n_epochs=self.n_epochs,
                                learning_rate=self.learning_rate,
                                init=self.init,
                                min_dist=self.min_dist,
                                spread=self.spread,
                                low_memory=self.low_memory,
                                set_op_mix_ratio=self.set_op_mix_ratio,
                                local_connectivity=self.local_connectivity,
                                repulsion_strength=self.repulsion_strength,
                                negative_sample_rate=self.negative_sample_rate,
                                transform_queue_size=self.transform_queue_size,
                                a=self.a,
                                b=self.b,
                                random_state=self.random_state,
                                angular_rp_forest=self.angular_rp_forest,
                                target_n_neighbors=self.target_n_neighbors,
                                target_metric=self.target_metric,
                                target_weight=self.target_weight,
                                transform_seed=self.transform_seed,
                                force_approximation_algorithm=self.force_approximation_algorithm,
                                verbose=self.verbose,
                                unique=self.unique).fit_transform(data)
        end = time.time()
        print('Adapted UMAP Layout time = %f (sec), per sample=%f (sec)' %
              (end - start_time, float(end - start_time) / N))

        return simple_umap

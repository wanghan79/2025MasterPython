# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
import numpy as np
from faiss import Kmeans as faiss_Kmeans
from tqdm import tqdm

DEFAULT_KMEANS_SEED = 1234

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, kmeans_labels, dataset, transform=None):

        self.imgs = self.make_dataset(image_indexes, kmeans_labels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, kmeans_labels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(kmeans_labels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset.imgs[idx]
            kmeans_label = label_to_idx[kmeans_labels[j]]
            noisy_label = dataset[idx][2]
            target = dataset[idx][3]
            domain_label = dataset[idx][4]
            images.append((path, kmeans_label, noisy_label, target, domain_label, idx))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel, target, domain_label, ind = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel, target, domain_label, ind

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D


def cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    kmeans_labels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        kmeans_labels.extend([cluster] * len(images))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])

    return ReassignedDataset(image_indexes, kmeans_labels, dataset, t)


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    # losses = faiss.vector_to_array(clus.obj)
    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False, centroids=None):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.
    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if i == 200 - 1:
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC(object):
    """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwidth of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        xb = preprocess_features(data)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0
    
    
class Kmeans(object):
    def __init__(
        self, k_list, data, epoch=0, init_centroids=None, frozen_centroids=False
    ):
        """
        Performs many k-means clustering.
        Args:
            data (np.array N * dim): data to cluster
        """
        super().__init__()
        self.k_list = k_list
        self.data = data
        self.d = data.shape[-1]
        self.init_centroids = init_centroids
        self.frozen_centroids = frozen_centroids

        self.logger = logging.getLogger("Kmeans")
        self.debug = False
        self.epoch = epoch + 1

    def compute_clusters(self):
        """compute cluster
        Returns:
            torch.tensor, list: clus_labels, centroids
        """
        data = self.data
        labels = []
        centroids = []

        tqdm_batch = tqdm(total=len(self.k_list), desc="[K-means]")
        for k_idx, each_k in enumerate(self.k_list):
            seed = k_idx * self.epoch + DEFAULT_KMEANS_SEED
            kmeans = faiss_Kmeans(
                self.d,
                each_k,
                niter=40,
                verbose=False,
                spherical=True,
                min_points_per_centroid=1,
                max_points_per_centroid=10000,
                gpu=True,
                seed=seed,
                frozen_centroids=self.frozen_centroids,
            )

            kmeans.train(data, init_centroids=self.init_centroids)

            _, I = kmeans.index.search(data, 1)
            labels.append(I.squeeze(1))
            C = kmeans.centroids
            centroids.append(C)

            tqdm_batch.update()
        tqdm_batch.close()

        labels = np.stack(labels, axis=0)

        return labels, centroids



def torch_kmeans(k_list, data, init_centroids=None, seed=0, frozen=False):
    if init_centroids is not None:
        init_centroids = init_centroids.cpu().numpy()
    km = Kmeans(
        k_list,
        data.cpu().detach().numpy(),
        epoch=seed,
        frozen_centroids=frozen,
        init_centroids=init_centroids,
    )
    clus_labels, centroids_npy = km.compute_clusters()
    clus_labels = torch.from_numpy(clus_labels).long().cuda()
    centroids = []
    for c in centroids_npy:
        centroids.append(torch.from_numpy(c).cuda())
    # compute phi
    clus_phi = []
    for i in range(len(k_list)):
        clus_phi.append(compute_variance(data, clus_labels[i], centroids[i]))

    return clus_labels, centroids, clus_phi


# variance


@torch.no_grad()
def compute_variance(
    data, cluster_labels, centroids, alpha=10, debug=False, num_class=None
):
    """compute variance for proto
    Args:
        data (torch.Tensor): data with shape [n, dim]
        cluster_labels (torch.Tensor): cluster labels of [n]
        centroids (torch.Tensor): cluster centroids [k, ndim]
        alpha (int, optional): Defaults to 10.
        debug (bool, optional): Defaults to False.
    Returns:
        [type]: [description]
    """

    k = len(centroids) if num_class is None else num_class
    phis = torch.zeros(k)
    for c in range(k):
        cluster_points = data[cluster_labels == c]
        c_len = len(cluster_points)
        if c_len == 0:
            phis[c] = -1
        elif c_len == 1:
            phis[c] = 0.05
        else:
            phis[c] = torch.sum(torch.norm(cluster_points - centroids[c], dim=1)) / (
                c_len * np.log(c_len + alpha)
            )
            if phis[c] < 0.05:
                phis[c] = 0.05

    if debug:
        print("size-phi:", end=" ")
        for i in range(k):
            size = (cluster_labels == i).sum().item()
            print(f"{size}[phi={phis[i].item():.3f}]", end=", ")
        print("\n")

    return phis
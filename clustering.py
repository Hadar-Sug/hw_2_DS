import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.01^2)
    """
    noise = np.random.normal(loc=0, scale=0.01, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    transformed_data = df[features]
    minimums = [min(transformed_data[0]), min(transformed_data[1])]
    sums = [sum(transformed_data[0]), sum(transformed_data[1])]
    for i in range(2):
        transformed_data[features[i]] = transformed_data[features[i]].apply(lambda x: (x - minimums[i]) / sums[i])
    return add_noise(transformed_data.to_numpy())


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """

    prev_centroids = choose_initial_centroids(data, k)

    while not np.array_equal(prev_centorids, current_centroids):
        assign_to_clusters(data, prev_centroids)

    # then we must update the coordinates of each centroid

    # centroids= np.array([[]])
    # for cent in centroids_loc:
    #     centroids = np.block(centroids, cent)

    # return labels, centroids


def choose_centroid(x, centroids):
    """
    iterate over the centroids and find the one that minimizes the distance
    :param x: point we're checking the distances to the centroids
    :param centroids: np array of all centroids (k,2)
    :return: coordinates of one of the centroids
    """
    min_dist_index = 0
    min_dist = np.linalg.norm(x - centroids[0])  # initialize with first centroid
    for i in range(centroids.shape[0]):
        this_dist = np.linalg.norm(x - centroids[i])
        if this_dist < min_dist:
            min_dist = this_dist
            min_dist_index = i
    return centroids[min_dist_index]


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    pass
    # plt.savefig(path)


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    pass
    # return distance


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    labels = np.array(data.shape)
    for i in range(data.shape[0]):  # till no changes occurred between following iterations
        labels[i] = choose_centroid(data[i], centroids)
    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    labels_df = pd.DataFrame(labels, columns=["first coord, second coord"])
    new_centroids = labels_df.groupby(['first coord', 'second coord']).mean()
    new_centroids = new_centroids.to_frame()
    return new_centroids.to_numpy()


def sort_centroids(centroids):
    return sorted(centroids, key=lambda row: np.linalg.norm(row))

def equal_centroids(prev, curr):
    prev_centroids = sort_centroids(prev)
    current_centroids = sort_centroids(curr)
    return np.array_equal(prev_centroids, current_centroids)

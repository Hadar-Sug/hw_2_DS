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
    minimums = [min(transformed_data[0]), min(transformed_data[1])]  # scaling params
    sums = [sum(transformed_data[0]), sum(transformed_data[1])]
    for i in range(2):  # actual scaling
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
    min_dist = dist(x, centroids[0])  # initialize with first centroid
    for i in range(centroids.shape[0]):
        this_dist = dist(x, centroids[i])
        if this_dist < min_dist:
            min_dist = this_dist
            min_dist_index = i
    return min_dist_index


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    # hope this works
    # edge case - what if the number of centroids dips below k
    labelled_data = np.append(data, labels, axis=1)  # add label for each datapoint (label column)
    labels_df = pd.DataFrame(labelled_data, columns=['first coord', 'second coord', 'centroid'])  # lets get it as a pds
    new_centroids = labels_df.sort_values('centroid').groupby(['centroid'])[
        'first coord', 'second coord'].mean()  # added sort values by centroid, i think this eliminates the need for sorting the centroids later
    new_centroids = new_centroids.to_frame()
    return new_centroids.to_numpy()


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """

    labelled_data = np.append(data, labels, axis=1)
    labels_df = pd.DataFrame(labelled_data, columns=['first coord', 'second coord', 'centroid'])
    grouped = labels_df.groupby(['centroid'])  # groupBy object
    groups_size = grouped.size(as_index=True)  # is it a list?
    grouped = grouped.to_frame().to_numpy()  # now numpy
    last = 0
    for size, centroid in zip(groups_size, centroids):
        rand_color = np.random.rand()
        plt.scatter(grouped[last:last + size, 0], grouped[last: last + size, 1], color=rand_color)
        plt.scatter(grouped[, 0], grouped[last: last + size, 1], color = rand_color)
        last = size
    plt.show()
    plt.savefig(path)


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    return np.sum((x - y) ** 2) ** 0.5  # check if this works


def sort_centroids(centroids):
    return centroids.sort(key=lambda row: dist(row, np.zeros(1, 2)))


def equal_centroids(prev, curr):
    prev_centroids = sort_centroids(prev)  # first we sort the centroids
    current_centroids = sort_centroids(curr)
    return np.array_equal(prev_centroids, current_centroids)  # then we compare

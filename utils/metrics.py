import torch
import matplotlib.pyplot as plt

def compute_pairwise_distances(particles):
    distances = torch.cdist(particles, particles)
    # get upper triangular part without diagonal
    n = particles.shape[0]
    indices = torch.triu_indices(n, n, offset=1)
    return distances[indices[0], indices[1]]

def plot_distance_distribution(distances, title="distance distribution"):
    plt.figure()
    plt.hist(distances.numpy(), bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel("distance")
    plt.ylabel("frequency")
    plt.show()

def compare_distributions(dist1, dist2, label1="original", label2="generated"):
    plt.figure()
    plt.hist(dist1.numpy(), bins=30, alpha=0.5, label=label1)
    plt.hist(dist2.numpy(), bins=30, alpha=0.5, label=label2)
    plt.title("distance comparison")
    plt.xlabel("distance")
    plt.ylabel("frequency")
    plt.legend()
    plt.show()

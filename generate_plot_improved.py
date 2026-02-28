import torch
import matplotlib.pyplot as plt
from models.vae import VAE
from data.generator import generate_particles
from utils.metrics import compute_pairwise_distances

def generate_improved_plot():
    n_particles = 50
    input_dim = n_particles * 3
    hidden_dim = 256
    latent_dim = 32

    # Load improved model
    model = VAE(input_dim, hidden_dim, latent_dim)
    try:
        model.load_state_dict(torch.load("model_improved.pth", weights_only=True))
        model.eval()
    except Exception as e:
        print("Could not load model_improved.pth. Please make sure train_improved.py ran.")
        return

    # Sample structures
    orig_pts = generate_particles(n_particles, min_distance=0.08)
    
    with torch.no_grad():
        z = torch.randn(1, latent_dim)
        reconstructed = model.decode(z)
        gen_pts = reconstructed.view(n_particles, 3)

    orig_dists = compute_pairwise_distances(orig_pts)
    gen_dists = compute_pairwise_distances(gen_pts)

    # Plotting
    fig = plt.figure(figsize=(12, 5))

    # Original 3D
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(orig_pts[:, 0], orig_pts[:, 1], orig_pts[:, 2], c='b', alpha=0.6)
    ax1.set_title("Original Structure")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_zlim([0, 1])

    # Generated 3D
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(gen_pts[:, 0].numpy(), gen_pts[:, 1].numpy(), gen_pts[:, 2].numpy(), c='g', alpha=0.6)
    ax2.set_title("Improved Generated")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_zlim([0, 1])

    # Distance distributions
    ax3 = fig.add_subplot(133)
    ax3.hist(orig_dists.numpy(), bins=30, alpha=0.5, label="Original", color='b')
    ax3.hist(gen_dists.numpy(), bins=30, alpha=0.5, label="Improved VAE", color='g')
    ax3.set_title("Pairwise Distances (Improved)")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("demo_plot_improved.png", dpi=150)
    print("Saved demo_plot_improved.png")

if __name__ == "__main__":
    generate_improved_plot()

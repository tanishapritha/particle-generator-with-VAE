import torch
import torch.nn as nn
import torch.optim as optim
from models.vae import VAE
from data.generator import generate_particles

def loss_function(reconstructed, x, mu, logvar, beta=0.1):
    mse = nn.functional.mse_loss(reconstructed, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Utilizing beta-VAE to emphasize reconstruction when beta < 1
    return mse + beta * kld

def train_improved():
    num_samples = 1500  # More samples for better learning
    n_particles = 50
    input_dim = n_particles * 3
    hidden_dim = 256    # Increased capacity
    latent_dim = 32     # Increased latent capacity
    epochs = 400
    learning_rate = 1e-3

    print("generating improved dataset...")
    # generate dataset of synthetic particle configurations
    dataset = []
    for _ in range(num_samples):
        pts = generate_particles(n_particles, min_distance=0.08)
        
        # KEY IMPROVEMENT: sorting particles by primary axis to break permutation symmetry.
        # This makes it *much* easier for a simple MLP VAE to learn the patterns.
        # We sort by X coordinate (and loosely by Y and Z)
        sorting_scores = pts[:, 0] * 100 + pts[:, 1] * 10 + pts[:, 2]
        indices = torch.argsort(sorting_scores)
        pts = pts[indices]
        
        dataset.append(pts)
    
    # flatten coordinates for model input
    dataset = torch.stack(dataset).view(num_samples, input_dim)

    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("starting improved training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # simple batching
        batch_size = 32
        for i in range(0, num_samples, batch_size):
            batch = dataset[i:i+batch_size]
            
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            loss = loss_function(reconstructed, batch, mu, logvar, beta=0.05)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch+1}/{epochs} | loss: {train_loss / num_samples:.4f}")

    torch.save(model.state_dict(), "model_improved.pth")
    print("saved improved model as model_improved.pth")

if __name__ == "__main__":
    train_improved()

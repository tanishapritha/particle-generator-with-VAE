import torch
import torch.nn as nn
import torch.optim as optim
from models.vae import VAE
from data.generator import generate_particles

def loss_function(reconstructed, x, mu, logvar):
    mse = nn.functional.mse_loss(reconstructed, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld

def train():
    num_samples = 500
    n_particles = 50
    input_dim = n_particles * 3
    hidden_dim = 128
    latent_dim = 16
    epochs = 100
    learning_rate = 1e-3

    print("generating dataset...")
    # generate dataset of synthetic particle configurations
    dataset = []
    for _ in range(num_samples):
        pts = generate_particles(n_particles, min_distance=0.08)
        dataset.append(pts)
    
    # flatten coordinates for model input
    dataset = torch.stack(dataset).view(num_samples, input_dim)

    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # simple batching
        batch_size = 32
        for i in range(0, num_samples, batch_size):
            batch = dataset[i:i+batch_size]
            
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            loss = loss_function(reconstructed, batch, mu, logvar)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"epoch {epoch+1}/{epochs} | loss: {train_loss / num_samples:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("saved model as model.pth")

if __name__ == "__main__":
    train()

import streamlit as st
import torch
import plotly.graph_objects as go
from models.vae import VAE
from utils.metrics import compute_pairwise_distances
from data.generator import generate_particles

st.title("3D Particle Structure Generator")

st.sidebar.header("Controls")
n_particles = st.sidebar.number_input("Number of particles", min_value=10, max_value=200, value=50)
latent_dim = st.sidebar.number_input("Latent dimension", min_value=2, max_value=64, value=16)
compare = st.sidebar.checkbox("Compare with training distribution")

if st.sidebar.button("Generate Structure"):
    input_dim = n_particles * 3
    hidden_dim = 128
    
    # load trained model
    model = VAE(input_dim, hidden_dim, latent_dim)
    try:
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
    except:
        st.warning("model.pth not found. Please train the model first.")
        st.stop()
        
    # sample latent vector
    with torch.no_grad():
        z = torch.randn(1, latent_dim)
        
        # generate new structure
        reconstructed = model.decode(z)
        generated_pts = reconstructed.view(n_particles, 3)
        
    # display results
    st.subheader("Interactive 3D Structure")
    
    fig = go.Figure(data=[go.Scatter3d(
        x=generated_pts[:, 0],
        y=generated_pts[:, 1],
        z=generated_pts[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue')
    )])
    
    fig.update_layout(scene=dict(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        zaxis=dict(range=[0, 1])
    ))
    st.plotly_chart(fig)
    
    st.subheader("Distance Distribution")
    from matplotlib import pyplot as plt
    
    gen_dists = compute_pairwise_distances(generated_pts)
    
    fig, ax = plt.subplots()
    if compare:
        orig_pts = generate_particles(n_particles, min_distance=0.08)
        orig_dists = compute_pairwise_distances(orig_pts)
        
        ax.hist(orig_dists.numpy(), bins=30, alpha=0.5, label="original")
        ax.hist(gen_dists.numpy(), bins=30, alpha=0.5, label="generated")
        ax.legend()
    else:
        ax.hist(gen_dists.numpy(), bins=30, alpha=0.7)
        
    ax.set_xlabel("distance")
    ax.set_ylabel("frequency")
    st.pyplot(fig)

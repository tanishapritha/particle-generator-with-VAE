# 3D Particle Structure Generator using VAE

Generates synthetic 3D particle configurations inside a unit cube, trains a simple Variational Autoencoder (VAE) to learn their geometric distribution, and allows interactive sampling of new structures.

## What it does:
- Generates synthetic 3D particle configurations
- Trains a small VAE
- Samples new structures from the latent space
- Visualizes structure and distance distribution
- Includes an interactive [Jupyter Notebook](explore.ipynb) connected to the project's venv for exploration

## Screenshots

### Baseline VAE
![Demo Plot](demo_plot.png)

### Improved VAE (Beta-VAE + Sorted Coordinates)
To dramatically improve how well the MLP learns the structure, we broke permutation symmetry by sorting particles during dataset generation and used a Beta-VAE loss weighting. The distributions align much better!
![Improved Demo Plot](demo_plot_improved.png)

## How to run:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Future work:
- Add stronger physical constraints
- Improve generative stability

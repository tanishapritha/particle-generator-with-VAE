# 3D Particle Structure Generator using VAE

Hey! Welcome to my project. I built a simple AI to generate 3D particles inside a box.

## What I built

I started by writing code to randomly place 50 particles in a 3D box, making sure they aren't completely on top of each other. That was my "original" dataset.

Then, I built a small Variational Autoencoder (VAE) from scratch using PyTorch. The goal was to see if the VAE could learn the hidden layout rules just from looking at the data, without me telling it the math!

### First attempt (Baseline VAE)
My first model learned to fit particles inside the box, but it struggled to space them out correctly. When you look at the histogram below, the generated (red) distances didn't really match the original (blue) distances.

![Demo Plot](demo_plot.png)

### The Improvement
I realized that because the particles were in a random order, it confused the AI. "Point 1" in one box was in a totally different spot than "Point 1" in another box. So, I grabbed the data, **sorted the particles by their coordinates**, and fed them back into a slightly bigger VAE model (I used a Beta-VAE approach to focus more on exact placements). 

The result? The new model figured out the structure! In the plot below, the AI-generated distances (green) beautifully trace the original layout distances.

![Improved Demo Plot](demo_plot_improved.png)

## Check it out

If you want to play with the interactive 3D graphs, you can run my Streamlit app:
```bash
python -m venv venv
source venv/Scripts/activate # or venv/bin/activate on Mac/Linux
pip install -r requirements.txt
python train.py
python train_improved.py
streamlit run app.py
```

I also included a [Jupyter Notebook](explore.ipynb) if you want to walk through the plotting and sampling code step-by-step.

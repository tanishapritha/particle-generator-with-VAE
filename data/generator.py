import torch
import matplotlib.pyplot as plt

def generate_particles(n_particles, min_distance=0.05):
    """
    generate random particles inside cube
    avoid particles being too close
    """
    particles = []
    
    while len(particles) < n_particles:
        new_particle = torch.rand(3)
        
        if len(particles) == 0:
            particles.append(new_particle)
            continue
            
        # calculate distances to existing particles
        existing = torch.stack(particles)
        distances = torch.linalg.norm(existing - new_particle, dim=1)
        
        if torch.all(distances >= min_distance):
            particles.append(new_particle)
            
    return torch.stack(particles)

if __name__ == "__main__":
    # test block to visualize one structure
    pts = generate_particles(50, min_distance=0.1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', marker='o')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    
    plt.show()

# ------------------------------------------------------------
# convergence.py
#
# This module is responsible for computing the global energy
# and plotting the convergence graph for the Markov-Vision project.
#
# Functions:
# - compute_energy(labels, beta):
#     Calculates the global energy of the label map using the
#     Potts model by counting mismatches between adjacent pixels.
#
# - plot_convergence(history):
#     Plots the energy history graph over the iterations to 
#     visually confirm model convergence.
#
# This file is used by the main application to evaluate model
# stability and provide the final validation visualization.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from config import BETA

def compute_energy(labels, beta=BETA):
    """
    Computes the global energy of the label map (Potts Model).
    Measures the sum of mismatches between neighboring pixels.
    
    Arguments:
    - labels : 2D ndarray (H, W) containing integers (pixel classes)
    - beta   : cohesion strength (float constant)
    
    Returns:
    - float : the global energy value
    """
    H, W = labels.shape
    
    # Horizontal differences: compare each pixel with its right neighbor
    mismatches_h = np.sum(labels[:, :-1] != labels[:, 1:])
    
    # Vertical differences: compare each pixel with its bottom neighbor
    mismatches_v = np.sum(labels[:-1, :] != labels[1:, :])
    
    # Global energy is the total number of mismatches multiplied by beta
    global_energy = beta * (mismatches_h + mismatches_v)
    
    return float(global_energy)

def plot_convergence(history):
    """
        Displays the convergence graph showing energy evolution over iterations.
        Dynamically adapts whether the history contains raw energy values
         or label matrices .

        Arguments:
        - history : list of energies (floats) OR list of label maps (ndarrays)
        """

    import matplotlib.ticker as ticker

    if len(history) > 0 and isinstance(history[0], np.ndarray):
        energy_values = [compute_energy(labels) for labels in history]
    else:
        energy_values = history

    total_iterations = len(energy_values)

    # Clean technical aesthetic with matching background palette color
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#f9feff')
    ax.set_facecolor('#ffffff')

    # Plotting a distinct line with elegant markers
    ax.plot(energy_values, color='#1b4f72', linewidth=2, marker='.', markersize=6, label="Global Energy")

    # Titles and axis formatting
    ax.set_title("Markov Random Field Model Convergence", fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel("Iterations ", fontsize=10, fontweight='semibold', labelpad=8, color='#34495e')
    ax.set_ylabel("Global Energy (Potts Cost)", fontsize=10, fontweight='semibold', labelpad=8, color='#34495e')

    # --- DYNAMIC GRID & TICK INTERVALS ---
    if total_iterations <= 20:
        # Perfect detail for standard runs: a grid line on EVERY single step
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        # Hide every second text label to avoid text collision, keeping lines clear
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)}' if x % 2 == 0 else ''))
    elif total_iterations <= 50:
        # Mid-range scale optimization: grid line and label every 5 steps
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    else:
        # Large scale automation defense: grid line and label every 10 steps
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10.0))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # Subtle, light dashed grid lines for both axes
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')

    # Legend formatting
    ax.legend(loc="upper right", frameon=True, facecolor='#ffffff', edgecolor='#e5e7e9')

    # Clean up the spines/borders
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#cccccc')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("    INDEPENDENT AUTOMATED TEST — convergence.py")
    print("=" * 60)
    
    # Test 1: Testing with a simulated history (before S3 is integrated)
    print("\n[Test 1] Launching graph with simulated history data...")
    fake_history = [500, 380, 270, 190, 140, 105, 82, 67, 58, 52,
                    48, 45, 43, 41, 40, 39, 38, 38, 37, 37]
    
    # Blocking GUI window for visual validation (close the window to proceed with tests)
    plot_convergence(fake_history)
    
    # Test 2: compute_energy calculation validation on dummy labels
    print("\n[Test 2] Validating mathematical energy calculations on mock labels...")
    fake_labels = np.random.randint(0, 3, (128, 128))
    e = compute_energy(fake_labels, beta=1.5)
    print('Initial energy (random noise) :', e)
    
    uniform_labels = np.zeros((128, 128), dtype=int)
    e2 = compute_energy(uniform_labels, beta=1.5)
    print('Uniform labels energy         :', e2)
    
    # Safety validation assertions
    assert e2 == 0.0, "ERROR: The energy of a perfectly uniform image must be exactly 0.0"
    assert e > e2, "ERROR: Random noise must yield higher energy than a uniform image"
    
    print("\n✔ If the curve decreases and stabilizes AND the uniform image energy is 0 -> your part is done.")
    print("[OK] Everything works perfectly! Your module is ready for the repository.")
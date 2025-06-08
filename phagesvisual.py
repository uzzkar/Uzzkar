import numpy as np          # For numerical operations
import pandas as pd         # For data manipulation and analysis
import time  # Import the time module to track the computation duration

import matplotlib.pyplot as plt  # For plotting graphs
from matplotlib.lines import Line2D  # Used for custom legend entries
import matplotlib.colors as mcolors
import seaborn as sns     

import random as rand
from numba import jit, njit  # JIT compiler for Python

def localizeseed_between(ax1, ax2, iterations, seed, p, tbad_list, fig):
    """
    Plot vertical grey lines between the two subplots, with intensity adjusted by bad event duration.
    
    Parameters:
    - ax1: The upper plot axis (lysogens).
    - ax2: The lower plot axis (free phages).
    - iterations: X-axis values (time).
    - seed: Array of seed values.
    - p: Threshold for determining when to plot a grey line.
    - tbad_list: List of bad event durations to adjust the intensity (darker for longer events).
    - fig: The overall figure, to allow drawing between subplots.
    """
    # Normalize the bad durations to control the intensity of the grey lines
    max_tbad = max(tbad_list)
    min_tbad = min(tbad_list)

    normalized_tbad = [
        0.15 + 0.8 * (tbad - min_tbad) / (max_tbad - min_tbad) if max_tbad != min_tbad else 0.4
        for tbad in tbad_list
    ]

    # Get the positions of the two axes in figure coordinates
    ax1_pos = ax1.get_position()
    ax2_pos = ax2.get_position()

    # Loop through the seed and plot grey lines between subplots with varying intensities
    for i, s in enumerate(seed):
        if s < p:
            x_pos = iterations[i]
            intensity = normalized_tbad[i]  # Use the normalized tbad to adjust intensity (alpha)

            # Transform the x-coordinate to figure space
            x_fig = ax1.transData.transform((x_pos, 0))[0]
            x_fig_norm = fig.transFigure.inverted().transform((x_fig, 0))[0]

            # Draw grey lines between the two subplots in figure coordinates
            fig.lines.append(plt.Line2D([x_fig_norm, x_fig_norm],
                                        [ax2_pos.y1, ax1_pos.y0],  # from the bottom of ax1 to the top of ax2
                                        transform=fig.transFigure, color='grey', lw=1.5, alpha=intensity))


def plot_2(ha, hb, hab, hfa, hfb, omega, p, tbad_list, xa, sa, xb, sb, scale, y_cutoff=9, seed=None):
    """
    """
    
    length = len(ha)
    iterations = np.arange(length)
    
    # Convert integer y_cutoff to actual cutoff for log scale
    y_cutoff_val = 10 ** (-y_cutoff)
    
    # Ensure no values are below the threshold for log scale plots
    if scale == 1:
        ha = np.clip(ha, y_cutoff_val, None)
        hb = np.clip(hb, y_cutoff_val, None)
        hab = np.clip(hab, y_cutoff_val, None)
    
    # Calculate averages for sorting the lysogen areas
    avg_ha, avg_hb, avg_hab = np.mean(ha), np.mean(hb), np.mean(hab)
    lysogens = [
        (avg_ha, ha, 'red', 'Lysogens A'),
        (avg_hb, hb, 'blue', 'Lysogens B'),
        (avg_hab, hab, 'green', 'Lysogens AB')
    ]
    lysogens_sorted = sorted(lysogens, key=lambda x: x[0], reverse=True)
    
    # Create the figure with subplots and share the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'hspace': 0.1})
    
    # Add Lysogen Concentration Plot
    for avg, data, color, label in lysogens_sorted:
        ax1.fill_between(iterations, 0, data, color=color, alpha=0.65, label=label)

    # If the scale flag is set, use a logarithmic scale for y-axis
    if scale == 1:
        ax1.set_yscale('log')
        ax1.set_ylim(y_cutoff_val, max(np.max(ha), np.max(hb), np.max(hab)))

    if scale == 2:
        ax1.set_yscale('log')
        ax1.set_ylim(y_cutoff_val, max(np.max(ha), np.max(hb), np.max(hab)))

        ax2.set_yscale('log')
        ax2.set_ylim(y_cutoff_val, 10)
        

    ax1.set_ylabel("Lysogens", fontsize=16)

    ax1.text(0.01, 0.91, f'$\\mathbf{{x_A= {xa:.2f}, \\sigma_A \\cdot \\tau = {sa * 1000:.1f}}}$', 
             transform=ax1.transAxes, fontsize=14, color='red', fontstyle='italic', fontweight='bold')
    ax1.plot([0.30, 0.35], [0.94, 0.94], color='red', lw=4, transform=ax1.transAxes)
    ax1.text(0.36, 0.91, 'A Lysogens', transform=ax1.transAxes, fontsize=14, color='red', fontweight='bold')
    
    ax1.text(0.01, 0.80, f'$\\mathbf{{x_B = {xb:.2f}, \\sigma_B \\cdot \\tau = {sb * 1000:.1f}}}$', 
             transform=ax1.transAxes, fontsize=14, color='blue', fontstyle='italic', fontweight='bold')
    ax1.plot([0.30, 0.35], [0.84, 0.84], color='blue', lw=4, transform=ax1.transAxes)
    ax1.text(0.36, 0.80, 'B Lysogens', transform=ax1.transAxes, fontsize=14, color='blue', fontweight='bold')


    # Free Phages Plot
    ax2.plot(iterations, hfa, linestyle='-', color='red', lw=2, label='Free Phages A')
    ax2.plot(iterations, hfb, linestyle='-', color='blue', lw=2, label='Free Phages B')

    # Adjust the position of the y-label with labelpad
    #ax2.set_ylabel("Free Phages", fontsize=16, labelpad=15)  # Move the label further away from the axis
    #ax2.set_xlabel("Events", fontsize=16)
    #ax2.legend(loc='upper right', fontsize=12)

    # Custom legend for Lysogens AB (green)
    ax1.legend([Line2D([0], [0], color='green', lw=4)], ['Lysogens AB'], loc='upper right', fontsize=12)

    # Configure ticks and axis properties
    ax1.set_xticks([])
    ax2.set_xticks([length // 4, length // 2, 3 * length // 4])
    ax1.tick_params(axis='both', labelsize=16, width=2)
    ax2.tick_params(axis='both', labelsize=16, width=2)

    # If seed is provided, add grey lines between subplots
    if seed is not None:
        localizeseed_between(ax1, ax2, iterations, seed, p, tbad_list, fig)

    plt.show()

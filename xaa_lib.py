import numpy as np          # For numerical operations
import pandas as pd         # For data manipulation and analysis
import time  # Import the time module to track the computation duration

# For visualizations
import matplotlib.pyplot as plt  # For plotting graphs
from matplotlib.lines import Line2D  # Used for custom legend entries
import seaborn as sns     
import random as rand
from numba import jit, njit  # JIT compiler for Python

@njit
def bad_event(a, b, ab, sa, sb, tbad):
    bad_duration = np.random.exponential(tbad)
    bad_duration = int(np.floor(bad_duration))  # Ensure integer outcome

    a *= (1 - sa) ** (bad_duration)
    b *= (1 - sb) ** (bad_duration)
    ab *= (1 - sa - sb + sa * sb) ** (bad_duration)

    fa = 0
    fb = 0
    return a, b, ab, fa, fb, bad_duration

@njit
def good_event(fa, fb, a, b, ab, o, xa, xaa, xb, xbb, sa, sb, omega):
    # Total density (rho)
    rho = a + b + ab + o ## Check
    delta = 0 ## Check
    fa += (a * sa + ab * sa * (1 - sb * sb / ( sa + sb + 10 ** -12))) * omega
    fb += (b * sb + ab * sb * (1 - sa * sa / ( sa + sb + 10 ** -12))) * omega
    a *= (1 - sa)
    b *= (1 - sb)
    ab *= (1 - sa - sb + sa * sb)
    FA = 1 - np.exp(-fa / (delta + rho)) ## Check
    Fsa = FA * np.exp(-fa / (delta + rho)) ## Check
    Fma = 1 - 2 * np.exp(-fa / (delta + rho)) + np.exp(-2 * fa / (delta + rho)) ## Check
    
    # Compute FB, Fsb, Fmb for B
    FB = 1 - np.exp(-fb / (delta + rho)) ## Check
    Fsb = FB * np.exp(-fb / (delta + rho)) ## Check
    Fmb = 1 - 2 * np.exp(-fb / (delta + rho)) + np.exp(-2 * fb / (delta + rho)) ## Check

    # Compute Plyt(A) and Plyt(B)
    Plyt_A = 1 - (xa * Fsa + xaa * Fma) / (FA + 1e-12) ## Check  # Avoid division by zero
    Plyt_B = 1 - (xb * Fsb + xbb * Fmb) / (FB + 1e-12) ## Check  # Avoid division by zero

    # Compute updated densities (a_star, b_star, ab_star, o_star)
    a_star = a + xa * o * Fsa * (1 - FB) - a * FB \
             + xaa * o * Fma * (1 - FB)  ## Check
    b_star = b + xb * o * Fsb * (1 - FA) - b * FA \
             + xbb * o * Fmb * (1 - FA)  ## Check
    ab_star = ab \
        + xa * b * Fsa + xaa * b * Fma \
        + xb * a * Fsb + xbb * a * Fmb \
        + Fsa * Fsb * o * xa * xb \
        + Fma * Fsb * o * xaa * xb \
        + Fsa * Fmb * o * xa * xbb \
        + Fma * Fmb * o * xaa * xbb      ## Check
    
    # Compute updated free phage concentrations (fa_star, fb_star)
    fa_star = (1 - xa) * o * Fsa * (1 - FB * Plyt_B * fb / (fa + fb + 10 ** (-12))) * omega \
              + (1 - xa) * b * Fsa * omega \
              + (1 - xaa) * o * Fma * (1 - FB * Plyt_B * fb / (fa + fb + 10 ** (-12))) * omega \
              + (1 - xaa) * b * Fma * omega ## Check
    fb_star = (1 - xb) * o * Fsb * (1 - FA * Plyt_A * fa / (fa + fb + 10 ** (-12))) * omega \
              + (1 - xb) * a * Fsb * omega \
              + (1 - xbb) * o * Fmb * (1 - FA * Plyt_A * fa / (fa + fb + 10 ** (-12))) * omega \
              + (1 - xbb) * a * Fmb * omega ## Check

    o_star = 1 - (a + ab + b) ## Check

    a, b, ab, o, fa, fb = a_star, b_star, ab_star, o_star, fa_star, fb_star  ## Check

    return a, b, ab, o, fa, fb



@njit
def xaavsxbb(xa, xaa, sa, xb, xbb, sb, p, omega, tbad, seed, cutoff, delta):
    delta = 0
    # Initial values
    a, b, ab = 0, 0, 0
    o = 1 - (a + b + ab)
    fa, fb = 1, 1
    
    # Pre-allocate arrays for storage
    ha, hb, hab, hfa, hfb = np.zeros(len(seed)), np.zeros(len(seed)), np.zeros(len(seed)), np.zeros(len(seed)), np.zeros(len(seed))
    tbad_list = np.zeros(len(seed), dtype=np.int32)  # To store bad event durations as integers

    # Death indicators
    a_death, b_death, eps = len(seed), len(seed), 1e-12
    siga = sa * sa / (sa + sb + eps)
    sigb = sb * sb / (sa + sb + eps)

    # Main simulation loop
    for i in range(len(seed)):
        
        if p > seed[i]:
            bad_duration = np.random.exponential(tbad)
            bad_duration = int(np.floor(bad_duration))  
            a *= (1 - sa) ** (bad_duration)
            b *= (1 - sb) ** (bad_duration)
            ab *= (1 - sa - sb + sa * sb) ** (bad_duration)
            tbad_list[i] = bad_duration  
            fa, fb = 0, 0
        if p < seed[i]:
            tbad_list[i] = 0  # No bad event
            a, b, ab, o, fa, fb = good_event(fa, fb, a, b, ab, o, xa, xaa, xb, xbb, sa, sb, omega)

        # Track historical values
        ha[i], hb[i], hab[i], hfa[i], hfb[i] = a, b, ab, fa, fb
        if (fb + b + ab) < cutoff and b_death == len(seed):
            b_death = i
            b, ab, fb = 0, 0, 0
        # Check if either agent has died and record the iteration
        if (fa + a + ab) < cutoff and a_death == len(seed):
            a_death = i
            a, ab, fa = 0, 0, 0
            
        

    # Return only the valid portion of each array
    return ha[:i+1], hb[:i+1], hab[:i+1], hfa[:i+1], hfb[:i+1], tbad_list[:i+1], a_death, b_death



#### Evolution algorithms and visual of thoose

@njit
def compute_iterations(
    xa, xaa, sa, sb, xb_values, xbb_values, iterations, cutoff, p, omega, tbad, steps, games, conf
):
    """
    Perform the iterative simulation to calculate xa and xaa evolution.
    """
    xa_evolution = np.zeros(steps)  # Pre-allocate array for xa evolution (x*)
    xaa_evolution = np.zeros(steps)  # Pre-allocate array for xaa evolution (x**)
    step_completed = 0

    for step in range(steps):
        # Initialize results
        results = []

        # Test each xb and xbb combination
        for xb in xb_values:
            for xbb in xbb_values:
                wins_a = 0
                wins_b = 0
                fights = 0

                # Simulate 100 fights for each combination
                for _ in range(games):
                    seed = np.random.rand(iterations)
                    _, _, _, _, _, _, a_death, b_death = xaavsxbb(
                        xa, xaa, sa, xb, xbb, sb, p, omega, tbad, seed, cutoff
                    )

                    # Count wins and fights
                    if a_death != iterations and b_death > a_death + 100:
                        wins_b += 1
                        fights += 1
                    elif b_death != iterations and a_death > b_death + 100:
                        wins_a += 1
                        fights += 1

                # Calculate win percentage for b
                win_percentage_b = wins_b / (fights + 1e-12)
                results.append((xb, xbb, win_percentage_b))

        # Convert results to NumPy array
        results = np.array(results, dtype=np.float64)

        # Find the best xb and xbb
        best_idx = np.argmax(results[:, 2])
        best_result = results[best_idx]

        # Check if the best agent's win percentage is greater than 55%
        if best_result[2] > conf:
            xa = best_result[0]
            xaa = best_result[1]

            # Track evolution
            xa_evolution[step] = xa * 100  # Convert to percentage for visualization
            xaa_evolution[step] = xaa * 100  # Convert to percentage for visualization

            step_completed += 1
        else:
            # Stop if no agent wins more than 55% of the time
            break

    return xa_evolution[:step_completed], xaa_evolution[:step_completed], step_completed


def plot_evolution(xa_evolution, xaa_evolution, step_completed, p, omega):
    """
    Plot the evolution of xa (x*) and xaa (x**).
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Plot xa evolution
    axs[0].plot(range(step_completed), xa_evolution, marker='o', color='blue', label='x*')
    axs[0].set_title(f"p = {p}, Ω = {omega}, δ = 1", fontsize=14)
    axs[0].set_ylabel("x* %", fontsize=12)
    axs[0].set_xlabel("Evolution", fontsize=12)
    axs[0].grid(True)
    axs[0].legend()

    # Plot xaa evolution
    axs[1].plot(range(step_completed), xaa_evolution, marker='s', color='red', label='x**')
    axs[1].set_ylabel("x** %", fontsize=12)
    axs[1].set_xlabel("Evolution", fontsize=12)
    axs[1].grid(True)
    axs[1].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def plot_last_50(xa_evolution, xaa_evolution, p, omega, transient):
    """
    Plot the last 50 values of xa (x*) and xaa (x**).
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot the last 50 values of xa_evolution
    axs[0].plot(range(len(xa_evolution[-transient:])), xa_evolution[-transient:], marker='o', color='blue', label='x*')
    axs[0].set_title(f"Last 50 Values: p = {p}, Ω = {omega}, δ = 1", fontsize=14)
    axs[0].set_ylabel("x* %", fontsize=12)
    axs[0].grid(True)
    axs[0].legend()

    # Plot the last 50 values of xaa_evolution
    axs[1].plot(range(len(xaa_evolution[-50:])), xaa_evolution[-50:], marker='s', color='red', label='x**')
    axs[1].set_ylabel("x** %", fontsize=12)
    axs[1].set_xlabel("Last 50 Steps", fontsize=12)
    axs[1].grid(True)
    axs[1].legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

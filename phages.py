import numpy as np          # For numerical operations
import pandas as pd         # For data manipulation and analysis
import time  # Import the time module to track the computation duration

# For visualizations
import matplotlib.pyplot as plt  # For plotting graphs
from matplotlib.lines import Line2D  # Used for custom legend entries
import seaborn as sns     

import random as rand
from numba import jit, njit  # JIT compiler for Python

## Defining periods
@njit
def good_event(a, o, b, ab, fa, fb, xa, xb, sa, sb, siga, sigb, omega):
    """
    Handles updates during a 'good' event.
    """
    # Total available space and cells
    rho = a + o + b + ab  # Total available space and cells

    

    # Phage activity against available free space
    Fa = 1 - 2.71828 ** (-fa / (rho))  # Area hit by phage A
    Fb = 1 - 2.71828 ** (-fb / (rho))  # Area hit by phage B

    # Lysogen updates based on phage hits
    a = a + xa * o * Fa * (1 - Fb) - a * Fb  # Lysogen A updates
    b = b + xb * o * Fb * (1 - Fa) - b * Fa  # Lysogen B updates
    ab = ab + a * Fb * xb + b * Fa * xa + o * Fa * Fb * xa * xb  # Combined lysogens

    # Free phage updates
    frac_a = fa / (fa + fb + 1e-9)  # Fraction of free phage A
    frac_b = fb / (fa + fb + 1e-9)  # Fraction of free phage B
    fa = (1 - xa) * o * Fa * (1 - (1 - xb) * Fb * frac_b) * omega + (1 - xa) * b * Fa * omega
    fb = (1 - xb) * o * Fb * (1 - (1 - xa) * Fa * frac_a) * omega + (1 - xb) * a * Fb * omega

    # Lysogens decay
    a *= (1 - sa)
    b *= (1 - sb)
    ab *= (1 - sa - sb + sb * sa)

    # Phages reproduce due to lysis
    fa += (sa * a + sa * (1 - sigb) * ab) * omega
    fb += (sb * b + sb * (1 - siga) * ab) * omega

    # Update available free space
    o = 1 - (a + b + ab)
    
    return a, o, b, ab, fa, fb


@njit
def bad_event(a, b, ab, sa, sb, tbad, omega):
    """
    Handles updates during a 'bad' event where cells decay and free phages die.
    Duration of the bad event follows an exponential distribution but is rounded to the nearest integer.
    """
    bad_duration = np.random.exponential(tbad)
    bad_duration = int(np.floor(bad_duration))  # Ensuring integer outcome

    a *= (1 - sa) ** (bad_duration + 1)
    b *= (1 - sb) ** (bad_duration + 1)
    ab *= (1 - sa - sb + sa * sb) ** (bad_duration + 1)
    fa = (sa * a + ab * sa * (1 - sb * sb / (sa + sb))) * omega
    fb = (sb * b + sb * ab * (1 - sa * sa / (sa + sb))) * omega

    return a, b, ab, fa, fb, bad_duration

### Changed
@njit
def simo(xa, sa, xb, sb, p, omega, tbad, seed, cutoff):
    """
    Standard simulation. Here there gets filled in free lysogens, and they dont grow. 
    """
    # Starting concentrations, Assuming we start with a good event
    a, b, ab, o= 0, 0, 0, 1
    fa, fb = 1, 1
    siga, sigb = sa * sa / (sa + sb + 10 ** (-12)), sb * sb / (sa + sb + 10 ** (-12)) 
    a, o, b, ab, fa, fb = good_event(a, o, b , ab , fa, fb, xa , xb , sa , sb , siga, sigb, omega)
    
    ## Storing information
    a_death_control, b_death_control = 0, 0
    a_death, b_death = len(seed), len(seed)
    # Storing results over time
    ha, hb, hab, hfa, hfb = np.zeros(len(seed)), np.zeros(len(seed)), np.zeros(len(seed)), np.zeros(len(seed)), np.zeros(len(seed))

    # List to store bad event durations
    tbad_list = []

    # Simulation loop over iterations
    for i in range(len(seed)):
        # Bad event
        if p > seed[i]:
            bad_duration = np.random.exponential(tbad)
            a *= (1 - sa) ** (bad_duration + 1)
            b *= (1 - sb) ** (bad_duration + 1)
            ab *= (1 - sa - sb + sa * sb) ** (bad_duration + 1)
            fa = (sa * a + ab * sa * (1 - sb * sb / (sa + sb))) * omega
            fb = (sb * b + sb * ab * (1 - sa * sa / (sa + sb))) * omega
            # Append the drawn value to tbad_list
            tbad_list.append(bad_duration)            

        # Good event
        if p < seed[i]:
            a, o, b, ab, fa, fb = good_event(a, o, b, ab, fa, fb, xa, xb, sa, sb, siga, sigb, omega)
            tbad_list.append(0)

        ## Control for death
        if a + fa + ab < cutoff:
            a, ab, fa = 0, 0, 0 
            a_death_control += 1
            if a_death_control < 1.1:
                a_death = i
                
        if b + fb + ab < cutoff:
            b, ab, fb = 0, 0, 0
            b_death_control += 1
            if b_death_control < 1.1:
                b_death = i
                
        # Store the updated values in the arrays
        ha[i], hb[i], hab[i], hfa[i], hfb[i] = a, b, ab, fa, fb
    return ha, hb, hab, hfa, hfb, tbad_list, a_death, b_death



def find_winning_and_strongest_agent(xa, sa, xb_values, sb_values, p, omega, tbad, cutoff, fights):
    """
    """
    
    def run_simulations(xa, xb, sa, sb, p, omega, tbad, cutoff, fights):
        # Initialize arrays to store death occurrences
        a_excluded = np.zeros(fights)
        b_excluded = np.zeros(fights)

        for i in range(fights):
            # Generate a random seed for this simulation
            seed = np.random.rand(1000 * 5)  # Seed size optimized for performance
            seed[0] = 1
            
            # Run the simulation
            ha, hb, hab, hfa, hfb, death_a, death_b = simod(xa, sa, xb, sb, p, omega, tbad, seed, cutoff)
            
            # Store whether agent A or B died
            a_excluded[i] = death_a
            b_excluded[i] = death_b

        # Count non-zero deaths for agent A and B
        nonzero_a = np.count_nonzero(a_excluded)
        nonzero_b = np.count_nonzero(b_excluded)
        
        # Reverse the winning criteria: 
        # Agent A wins when Agent B dies, and Agent B wins when Agent A dies
        report = {
            "a_win_percentage": round((nonzero_b / fights) * 100, 2),  # Agent A wins if B dies
            "b_win_percentage": round((nonzero_a / fights) * 100, 2),  # Agent B wins if A dies
        }
        
        return report

    def test_agents_against_a(xa, sa, xb_values, sb_values, p, omega, tbad, cutoff, fights):
        results = []  # To store the results for each (xb, sb) pair
        pairs = []  # To store the tested pairs of (xb, sb)

        for xb in xb_values:
            for sb in sb_values:
                # Run simulations for the current pair of (xb, sb)
                report = run_simulations(xa, xb, sa, sb, p, omega, tbad, cutoff, fights)
                
                # Store the pair and the result
                pairs.append((xb, sb))
                results.append((report['a_win_percentage'], report['b_win_percentage']))

        return np.array(pairs), np.array(results)
    
    def find_winning_agents_against_a(pairs, results):
        # Find the agents (pairs of xb and sb) where Agent B won more than 50% of the time
        winning_agents = []
        for i, result in enumerate(results):
            if result[1] > 50:  # Agent B win percentage > 50%
                winning_agents.append((pairs[i][0], pairs[i][1], result[1]))  # (xb, sb, b_win_percentage)
        return winning_agents

    def find_strongest_agent(pairs, results):
        # Extract only the win percentages for Agent B
        b_win_percentages = np.array([result[1] for result in results])
        
        # Find indices where Agent B won more than 50%
        valid_indices = np.where(b_win_percentages > 50)[0]
        
        # If no valid agents are found, return None
        if len(valid_indices) == 0:
            return None, None
        
        # Find the index of the maximum win percentage from the valid indices
        max_index = valid_indices[np.argmax(b_win_percentages[valid_indices])]
        
        # Return the best agent (xb, sb) and the corresponding win percentage
        best_agent = pairs[max_index]
        max_win_percentage = b_win_percentages[max_index]
        
        return best_agent, max_win_percentage
    print("Agent A:", "Xa =",xa, "Sigma A =",sa)
    # Run the simulations
    pairs, results = test_agents_against_a(xa, sa, xb_values, sb_values, p, omega, tbad, cutoff, fights)
    
    # Find all agents that won more than 50% of the time against Agent A
    winning_agents = find_winning_agents_against_a(pairs, results)
    
    # Find the strongest agent that won the most against Agent A
    best_agent, max_win_percentage = find_strongest_agent(pairs, results)

    # Output results
    print("\nAgents that won more than 50% of the time against Agent A:")
    for agent in winning_agents:
        xb, sb, win_percentage = agent
        print(f"Agent B with (xb={xb:.4f}, sb={sb:.5f}) won {win_percentage}% of the time against Agent A.")
    
    # Print the strongest agent
    if best_agent is not None:
        xb, sb = best_agent
        print(f"\nAgent B with (xb={xb:.4f}, sb={sb:.5f}) won the most, with a win percentage of {max_win_percentage}%.")
    else:
        print("No agent won more than 50% of the time against Agent A.")
    
    return winning_agents, best_agent




@njit
def simod(xa, sa, xb, sb, p, omega, tbad, seed, cutoff):
    """
    
    """

    ## Initialize variables and arrays
    iterations = len(seed)
    
    a = 0
    b = 0
    ab = 0
    o = 1 - (a + b + ab)
    fa = 1
    fb = 1

    ha = np.zeros(iterations)
    hb = np.zeros(iterations)
    hab = np.zeros(iterations)
    hfa = np.zeros(iterations)
    hfb = np.zeros(iterations)

    death_a = 0
    death_b = 0

    eps = 10 ** (-12)
    siga = sa * sa / (sa + sb + eps)
    sigb = sb * sb / (sa + sb + eps)

    i = 0
    while True:
        if i == iterations:
            break
        
        fraca = fa / (fa + fb + eps)
        fracb = fb / (fa + fb + eps)

        if p > seed[i]:
            bad_duration = np.random.exponential(tbad)
            a *= (1 - sa) ** (bad_duration + 1)
            b *= (1 - sb) ** (bad_duration + 1)
            ab *= (1 - sa - sb + sa * sb) ** (bad_duration + 1)
            fa = (a * sa + ab * sa * (1 - sb * sb / (sa + sb))) * omega
            fb = (b * sb + ab * sb * (1 - sa * sa / (sa + sb))) * omega
            
        else:
            rho = a + o + b + ab
            Fa = 1 - 2.71828 ** (-fa / (rho))
            Fb = 1 - 2.71828 ** (-fb / (rho))

            a = a + xa * o * Fa * (1 - Fb) - a * Fb
            b = b + xb * o * Fb * (1 - Fa) - b * Fa
            ab = ab + a * Fb * xb + b * Fa * xa + o * Fa * Fb * xa * xb

            fa = (1 - xa) * o * Fa * (1 - Fb * fracb) * omega + (1 - xa) * b * Fa * omega
            fb = (1 - xb) * o * Fb * (1 - Fa * fraca) * omega + (1 - xb) * a * Fb * omega

            a *= (1 - sa)
            b *= (1 - sb)
            ab *= (1 - sa - sb + sb * sa)

            fa += (sa * a + sa * (1 - sigb) * ab) * omega
            fb += (sb * b + sb * (1 - siga) * ab) * omega

            o = 1 - (a + b + ab)

        if fa + a + ab and death_a == 0:
            fa = 0
            a = 0
            death_a = i

        if fb + b + ab < cutoff and death_b == 0:
            fb = 0
            b = 0
            death_b = i

        ha[i] = a
        hb[i] = b
        hab[i] = ab
        hfa[i] = fa
        hfb[i] = fb

        if death_a or death_b:
            break

        i += 1

    return ha, hb, hab, hfa, hfb, death_a, death_b

@njit
def simod1000(xa, sa, xb, sb, p, omega, tbad, seed, cutoff):
    """
    """

    ## Initialize variables and arrays
    iterations = len(seed)
    delta = 0
    a = 0
    b = 0
    ab = 0
    o = 1 - (a + b + ab)
    fa = 1
    fb = 1
    b_trigger = 0
    a_trigger = 0
    min = 100 ## extra run
    ha = np.zeros(iterations + 100)  # Extend arrays to accommodate additional 1000 iterations
    hb = np.zeros(iterations + 100)
    hab = np.zeros(iterations + 100)
    hfa = np.zeros(iterations + 100)
    hfb = np.zeros(iterations + 100)

    death_a = len(seed)
    death_b = len(seed)
    extra_iterations = 0  # To track additional 1000 iterations after death

    eps = 10 ** (-12)
    siga = sa * sa / (sa + sb + eps)
    sigb = sb * sb / (sa + sb + eps)
    
    i = 0

    a_trigger = 0
    b_trigger = 0
    while True:
        if i == iterations + extra_iterations:
            break
        
        fraca = fa / (fa + fb + eps)
        fracb = fb / (fa + fb + eps)

        if p > seed[i % iterations]:  # Ensure seed usage stays within the original size
            bad_duration = np.random.exponential(tbad)
            a *= (1 - sa) ** (bad_duration + 1)
            b *= (1 - sb) ** (bad_duration + 1)
            ab *= (1 - sa - sb + sa * sb) ** (bad_duration + 1)
            fa = (a * sa + sa * ab * (1 - sb * sb / (sa + sb + eps))) * omega
            fb = (b * sb + sb * ab * (1 - sa * sa / (sa + sb + eps))) * omega
        else:
            rho = a + o + b + ab
            Fa = 1 - 2.71828 ** (-fa / (rho))
            Fb = 1 - 2.71828 ** (-fb / (rho))

            a = a + xa * o * Fa * (1 - Fb) - a * Fb
            b = b + xb * o * Fb * (1 - Fa) - b * Fa
            ab = ab + a * Fb * xb + b * Fa * xa + o * Fa * Fb * xa * xb

            fa = (1 - xa) * o * Fa * (1 - Fb * fracb * (1 - xb)) * omega + (1 - xa) * b * Fa * omega
            fb = (1 - xb) * o * Fb * (1 - Fa * fraca * (1 - xa)) * omega + (1 - xb) * a * Fb * omega

            a *= (1 - sa)
            b *= (1 - sb)
            ab *= (1 - sa - sb + sb * sa)

            fa += (sa * a + sa * (1 - sigb) * ab) * omega
            fb += (sb * b + sb * (1 - siga) * ab) * omega

            o = 1 - (a + b + ab)

        if fa + a < cutoff + a_trigger:
            death_a = i
            a = 0
            fa = 0
            ab = 0
            a_trigger = 2
        if fb + b < cutoff + b_trigger:
            death_b = i
            b = 0
            fb = 0
            b_trigger = 2

        if death_a < 100:
            break

        if death_b < 100:
            break
            
        ha[i] = a
        hb[i] = b
        hab[i] = ab
        hfa[i] = fa
        hfb[i] = fb

        # Break after 1000 iterations following the death of a phage
        if extra_iterations and (i >= death_a + min or i >= death_b + min):
            break

        i += 1

    return ha, hb, hab, hfa, hfb, death_a, death_b





@njit
def find_winning_and_strongest_agent_v(xa, sa, xb_values, sb_values, p, omega, tbad, cutoff, fights):
    """
    Modified function to return all win rates for visualization.
    """

    #@njit
    def run_simulations(xa, xb, sa, sb, p, omega, tbad, cutoff, fights):
        # Initialize arrays to store death occurrences
        a_win = 0
        b_win = 0
        real_fights = 0
        eps = 10 ** (-12)
        transient = 10
        for i in range(fights):
            # Generate a random seed for this simulation
            seed = np.random.rand(1000 * 10)  # Seed size optimized for performance
            seed[0] = 1

            # Run the simulation (replace this with your real simulation function)
            ha, hb, hab, hfa, hfb, death_a, death_b = simod1000(xa, sa, xb, sb, p, omega, tbad, seed, cutoff)

            ### RECODE MAYBE, it is okay if the one survives all after 10 iterations...
            if death_a > transient and death_b > transient:
                real_fights += 1
                if death_a > death_b:
                    a_win += 1
                    real_fights += 1
                if death_b > death_a:
                    b_win += 1
                    real_fights += 1
                if death_a == death_b:
                    a_win += 0.5
                    b_win += 0.5
                    real_fights += 1
        # Calculate win percentages for both agents
        a_win_percentage = (a_win / (real_fights + eps)) * 100  # Agent A wins if B dies
        b_win_percentage = (b_win / (real_fights + eps)) * 100  # Agent B wins if A dies

        return a_win_percentage, b_win_percentage


    def test_agents_against_a(xa, sa, xb_values, sb_values, p, omega, tbad, cutoff, fights):
        results = np.zeros((len(xb_values), len(sb_values)))  # To store results for each (xb, sb) pair

        for i in range(len(xb_values)):
            xb = xb_values[i]
            for j in range(len(sb_values)):
                sb = sb_values[j]
                if np.isclose(xa, xb) and np.isclose(sa, sb):
                    # If B agent is the same as A agent, set win rate to 50%
                    results[i, j] = 50.0
                else:
                    # Run simulations for the current pair of (xb, sb)
                    a_win_percentage, b_win_percentage = run_simulations(xa, xb, sa, sb, p, omega, tbad, cutoff, fights)
                    results[i, j] = b_win_percentage  # Store win percentage for Agent B

        return results

    # Run the simulations and calculate win rates for all agents
    win_rates = test_agents_against_a(xa, sa, xb_values, sb_values, p, omega, tbad, cutoff, fights)

    return win_rates
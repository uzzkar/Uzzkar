
# Phage Interaction Simulations

This repository contains simulation notebooks and supporting libraries for exploring competitive dynamics between bacteriophage strains under environmental stress and cross-immunity constraints.

## Overview

The simulations fall into three core categories:

1. **Single Trajectory Analyses**
   Most recomended to test out; Two phages competing against each other in stochastic simulation. `phages.py` and `phagesvisual.py`are needed besiedes build in liberarys.
   Focused simulations that track the rise and fall of individual phage strains, including their free phages and lysogen populations. Useful for detailed understanding of extinction and survival mechanisms.
   
3. **Flux Simulations**  
   Stochastic simulations of population dynamics in a continuous environment. These models capture phage and lysogen behavior across time under randomly occurring bad events.

4. **Cross-Immunity Heatmaps**  
   Simulations of pairwise competition between phage strategies (`x*`, `x**`) under varying immunity configurations. Results are visualized as heatmaps that show regions of evolutionary advantage.

## Structure

- `phages.py`: Core simulation logic, including stochastic updates of phage populations and bad/good event transitions.
- `phagesvisual.py`: Visualization utilities for time-series plots of lysogens and phages.
- `xaa_lib.py`: Tools for simulating pairwise evolutionary competition, computing dominance, and tracking cross-immunity heatmaps.

## Notebooks

- **GitHub_ xaa vs xbb IMMUNE and Heatmap Imune.ipynb**  
  Generates heatmaps of cross-immunity interactions between competing phage strategies.

- **GitHub_ Phage Single Trajectories and Death tracking.ipynb**  
  Tracks single phage lineage dynamics and extinction times under stress.

- **Githhub_ Flux Simulations GRID.ipynb**  
  Simulates full-system evolution with stochastic environmental flux, comparing multiple strategies in continuous time.

## Usage

1. Clone the repository.
2. Install requirements:  
   ```bash
   pip install numpy matplotlib seaborn numba pandas
   ```
3. Open and run any of the Jupyter notebooks to reproduce the figures and simulations.

## Authors

- Kim Sneppen
- Oskar Lund


## License

This project is released for academic and research use.

# PyTorch implementation of Bayesian neural network experiments for VOGN.

This directory contains the PyTorch implementation of the Bayesian neural network experiments for the VOGN convergence results in the paper.

## Requirements
The experiment was performed using Python 3.5.2 with the following Python packages:
* [torch](https://pytorch.org/) == 0.4.0
* [numpy](http://www.numpy.org/) == 1.14.1

## Scripts explanation:
There are 3 python scripts:
* `run_experiments.py` - This is the main script to run experiments.
* `experiments.py` - This script implements experiment classes for VOGN, Vadam and BBVI.
* `produce_plots.py` - This script is for producing the three leftmost plots in Figure 3 in the paper. It loads files produced by `run_experiments.py`.
* `produce_gifs.py` - This script is for producing the animated GIFs. It loads files produced by `run_experiments.py`.

### How to run experiments:
Experiments can be run using `run_experiments.py`.
This script will store its results in `/results/`.
The final results obtained from our run are provided in `/results/results.zip`.
By extracting the contents of this file to `/results/`, the plots in the paper can be reproduced exactly.

### How to plot the results:
The plots can be produced using `produce_plots.py`.
This script loads results from `/results/` and saves the plots as pdfs in `/plots/`. The animated GIFs can be produced using `produce_gifs.py`. This scripts loads results from `/results/` and saves the animated GIFs in `/animations/`.
By extracting `/results/results.zip`, which contains results obtained from our run, the plots can be reproduced exactly.

# PyTorch implementation of Bayesian neural network experiments on MNIST.

This directory contains the PyTorch implementation of the Bayesian neural network experiments on the MNIST dataset. This is an additional result that is not in the paper as it was performed later.

The experiments show the convergence speed and performance of Vadam compared to BBVI+Adam. We observe a general tendency of BBVI being negatively affected by noise, whereas Vadam seems much more robust, showing stable convergence even in noisy settings. Moreover, when BBVI converges, the Vadam and BBVI tends to perform very similarly.

## Requirements
The experiment was performed using Python 3.5.2 with the following Python packages:
* [torch](https://pytorch.org/) == 0.4.0
* [numpy](http://www.numpy.org/) == 1.14.1

## Scripts explanation:
There are 4 python scripts:
* `run_experiments.py` - This is the main script to run experiments.
* `experiments.py` - This script implements experiment classes for Vadam and BBVI.
* `produce_gifs.py` - This script is for producing the animated GIFs. It loads files produced by `run_experiments.py`.
* `visualize_results.ipynb` - This is a jupyter notebook for visualizing the results of the experiment. It loads files produced by `run_experiments.py`.

### How to run experiments:
Experiments can be run using `run_experiments.py`.
This script will store its results in `/results/`.
The final results obtained from our run are provided in `/results/results.zip`.
By extracting the contents of this file to `/results/`, the plots in the paper can be reproduced exactly.

### How to plot the results:
The results can be visualized using the notebook `visualize_results.ipynb`.
This notebook loads results from `/results/` and allow you to explore the results using widgets.
The animated GIFs can be produced using `produce_gifs.py`.
This scripts loads results from `/results/` and saves the animated GIFs in `/animations/`.
By extracting `/results/results.zip`, which contains results obtained from our run, the plots can be reproduced exactly.

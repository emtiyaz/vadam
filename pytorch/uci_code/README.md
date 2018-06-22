# PyTorch implementation of Bayesian neural network experiments on UCI datasets. 

This directory contains the PyTorch implementation of the Bayesian neural network experiments on the UCI datasets in the paper.

## Requirements
The experiment was performed using Python 3.5.2 with the following Python packages:
* [torch](https://pytorch.org/) == 0.4.0
* [numpy](http://www.numpy.org/) == 1.14.1
* [scipy](https://www.scipy.org/) == 1.0.0
* [sklearn](http://scikit-learn.org/stable/index.html) == 0.19.1
* [bayes_opt](https://github.com/fmfn/BayesianOptimization) == 0.6.0

## Scripts explanation:
There are 6 python scripts:
* `run_experiments.py` - This is the main script to run experiments.
* `experiments.py` - This script implements experiment classes for Vprop, Vadam and BBVI.
* `experiments_cv.py` - This script implements cross-validation experiment classes for Vadam and BBVI.
* `bayesopt.py` - This script implements the Bayesian optimization used for running the experiments.
* `produce_table.py` - This script is for producing Table 1 in the paper. It loads files produced by `run_experiments.py`.
* `produce_plots.py` - This script is for producing Figure 4 in the paper. It loads files produced by `run_experiments.py`.

### How to run experiments:
Experiments can be run using `run_experiments.py`. 
This script will store its results in `/results/`.
The final results obtained from our run are provided in `/results/results.zip`. 
By extracting the contents of this file to `/results/`, the table and plots in the paper can be reproduced exactly.

### How to produce the table:
The table can be produced using `produce_table.py`. 
This script loads results from `/results/` and outputs numbers used to produce the table.
By extracting `/results/results.zip`, which contains results obtained from our run, the table can be reproduced exactly.

### How to plot the results:
The plots can be produced using `produce_plots.py`. 
This script loads results from `/results/` and saves the plots as pdfs in `/plots/`.
By extracting `/results/results.zip`, which contains results obtained from our run, the plots can be reproduced exactly.


## Remarks
### Reproducibility of the result:
The result is reproducible up to some extent as the random seed for the Bayesian optimization was not fixed.
Moreover, the results in the paper were obtained using [Nvidia Tesla V100](https://www.nvidia.com/en-us/data-center/tesla-v100/) GPUs, which uses mixed-precision [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensorcore/).
This can further give slightly different results compared to those obtained using regular cores.
To allow for exact reproducibility, we have provided the results obtained from our run in `/results/results.zip`.

### Mistake in Appendix J:
In Appendix J of the ICML paper, it is stated that the initial precision of the variational distribution `q` was set to 10 for Vadam, which was a mistake.
This initial precision of `q` was actually set to the value of the precision of the prior distribution.
This ensures that the initial value of `s` in Algorithm 1 is set to 0, thus better matching the Adam optimizer (where the initial `s` is also set to 0).

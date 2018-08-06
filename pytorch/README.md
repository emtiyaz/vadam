# Install

In the folder containing `setup.py`, run
```
pip install --user -e .
```
The `--user` option ensures the library will only be installed for your user.
The `-e` option makes it possible to modify the library, and modifications will be loaded on the fly.

You should now be able to use it.

# Testing the Code

Some simple examples of how to use the code can be found in `examples`. Here you can see the switch from the Adam optimizer to the Vadam optimizer called on the same model.

# Reproducing UCI Experiments

The code for the UCI experiments together with obtained results can be found in `uci_code`. Using these results you should be able to reproduce these figures:

<img src="uci_code/plots/uci_legend-page-001.jpg" width="400">

<img src="uci_code/plots/uci_rmse_boston-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_concrete-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_energy-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_kin8nm-page-001.jpg" width="200">

<img src="uci_code/plots/uci_rmse_naval-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_powerplant-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_wine-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_yacht-page-001.jpg" width="200">

# Reproducing VOGN Experiments

The code for the VOGN experiments together with obtained results can be found in `vogn_code`. Using these results you should be able to reproduce these figures:

<img src="vogn_code/plots/plot_bs1_mc1-page-001.jpg" width="200"><img src="vogn_code/plots/plot_bs1_mc16-page-001.jpg" width="200"><img src="vogn_code/plots/plot_bs128_mc16_legend-page-001.jpg" width="200">

# MNIST Experiments

Code for MNIST experiments (not in the paper) together with obtained results can be found in `mnist_code`. These additional experiments show performance and convergence speed of Vadam compared to BBVI+Adam on various on MNIST. Here is an excerpt of the results, showing convergence for the two algorithms for different prior precisions, which is also used as the initial precision for the variational distribution:

<img src="mnist_code/animations/layer1_batchsize1.gif" width="600">

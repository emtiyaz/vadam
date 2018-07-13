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

The code for the UCI experiments together with obtained results can be found in `uci_code`. Using these results you should be able to reprodoce these figures:

<img src="uci_code/plots/uci_legend-page-001.jpg" width="200">

<img src="uci_code/plots/uci_rmse_boston-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_concrete-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_energy-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_kin8nm-page-001.jpg" width="200">

<img src="uci_code/plots/uci_rmse_naval-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_powerplant-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_wine-page-001.jpg" width="200"><img src="uci_code/plots/uci_rmse_yacht-page-001.jpg" width="200">

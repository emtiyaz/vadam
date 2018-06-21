import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#from https://tonysyu.github.io/plotting-error-bars.html#.WRwXWXmxjZs
def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None, pltcmd=None, linewidth = None, label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, pltcmd, color=color, linewidth = linewidth, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


#path = "./result/HalfCheetah-v1"
path = "./result/HalfCheetah-v1_2nd"

seed_list = [0, 1, 2, 3, 4]
seed_length = len(seed_list)
    
########################### Vadam  ######################################
vadam = 0
for i in seed_list:
    vadam += np.load(os.path.join(path, 'VADAM/vadam_precinit0_minprec10000_lr0.000100_gammaII0.999000_gammaI0.900000_seed%d_test_reward.npy' % i))
vadam = vadam / seed_length
vadam_error = np.std(vadam, axis=1)
vadam = np.mean(vadam, axis=1)

########################### Vadagrad  ######################################
vadagrad = 0
for i in seed_list:
    vadagrad += np.load(os.path.join(path, 'VADAGRAD/vadagrad_precinit10000_lr0.01_gammaII0.99_seed%d_test_reward.npy' % i))
vadagrad = vadagrad / seed_length
vadagrad_error = np.std(vadagrad, axis=1)
vadagrad = np.mean(vadagrad, axis=1)

########################### Adam ############################################
adam = 0
for i in seed_list:
    adam += np.load(os.path.join(path, 'ADAM/adam_lr0.0001_seed%d_test_reward.npy' % i))
adam = adam/seed_length
adam_error = np.std(adam, axis=1)
adam = np.mean(adam, axis=1)

########################## Noise+Adam ############################################
noise_adam = 0
for i in seed_list:
    noise_adam += np.load(os.path.join(path, 'NOISE+ADAM/noise+adam_precinit10000_lr0.0001_lrs0.01_seed%d_test_reward.npy' % i))
noise_adam = noise_adam/seed_length
noise_adam_error = np.std(noise_adam, axis=1)
noise_adam = np.mean(noise_adam, axis=1)

########################### SGD ############################################
sgd = 0
for i in seed_list:
    sgd += np.load(os.path.join(path, 'SGD/sgd_lr0.0001_seed%d_test_reward.npy' % i))
sgd = sgd/seed_length
sgd_error = np.std(sgd, axis=1)
sgd = np.mean(sgd, axis=1)

########################## SGD + NOISE######################################
sn = 0
for i in seed_list:
    sn += np.load(os.path.join(path, 'NOISE+SGD/noise+sgd_precinit10000_lr0.0001_lrs0.01_seed%d_test_reward.npy' % i))
sn = sn/seed_length
sn_error = np.std(sn, axis=1)
sn = np.mean(sn, axis=1)

##############################################################################

linewidth = 2
x = np.arange(300)*10 

""" Figure of all methods in the appendix """
f = plt.figure(figsize=(9,7))

errorfill(label="Vadam", x = x, y = vadam, yerr=vadam_error, color="r", pltcmd="-r", linewidth=linewidth)
        
errorfill(label="VadaGrad", x = x, y = vadagrad, yerr=vadagrad_error, color="b", pltcmd="-b", linewidth=linewidth)
        
errorfill(label="SGD-Plain", x = x, y = sgd, yerr=sgd_error, color="c", pltcmd=":c", linewidth=linewidth)
        
errorfill(label="SGD-Explore",    x = x, y = sn, yerr=sn_error, color="g", pltcmd="--g", linewidth=linewidth)
        
errorfill(label="Adam-Plain", x = x, y = adam, yerr=adam_error, color="m", pltcmd=":m", linewidth=linewidth)
        
errorfill(label="Adam-Explore",    x = x, y = noise_adam, yerr=noise_adam_error, color="k", pltcmd="--k", linewidth=linewidth)        
        
fontsize = 14
plt.legend(prop={"size":fontsize}, loc="lower right", ncol=3) 
plt.xlabel('Time step (x1000)', fontsize=fontsize)
plt.ylabel('Cumulative reward', fontsize=fontsize)
plt.xlim([-1, 3001])
plt.ylim([-1000, 5000])
plt.grid()

fig_name = path + ".pdf"
f.savefig(fig_name, bbox_inches='tight')

plt.show()



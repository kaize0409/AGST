import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import matplotlib
from matplotlib import rc, rcParams
font = {'size':14}

rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# generate similarted data
sns.set()
sns.set_style("whitegrid")
N = 20
x = np.arange(N)
mean_1 = 25 + np.random.normal(0.1, 1, N).cumsum()
std_1 = 2 + np.random.normal(0, .08, N).cumsum()

mean_2 = 15 + np.random.normal(0.2, 1, N).cumsum()
std_2 = 3 + np.random.normal(0, .1, N).cumsum()



font1 = {'family': 'Verdana',
             'weight': 'normal',
             'size': 18.5,
             }
font2 = {'family': 'Verdana',
         'weight': 'normal',
         'size': 18.5,
         }
plt.figure(figsize=(10, 7))
plt.xlabel(r'Length of Sequence ', fontdict=font1)
plt.ylabel('Hit Rate@20 (%)', fontdict=font1)

tic = [str(i) for i in x]
tic[-1] = r'$\geq15$'
tic[0] = r'$\leq3$'
#plt.xticks(tic)
plt.xticks(x, tic)


plt.plot(x, mean_1, 'r-', label='Core Users')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='r', alpha=0.2)

plt.plot(x, mean_2, 'b-', label='Casual Users')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='b', alpha=0.2)

plt.xticks(fontsize=16.5)
plt.tick_params(axis='y', labelsize=16.5)

plt.legend(prop=font2)

#plt.savefig('Amazon_book_performance.pdf',bbox_inches='tight')
plt.show()
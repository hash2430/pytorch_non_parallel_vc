import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4
baseline = (6.09, 6.22, 6.11, 6.14)
proposed = (6.08, 6.2, 6.08, 6.14)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity=0.8

rects1 = plt.bar(index, baseline, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Baseline')

rects2 = plt.bar(index + bar_width, proposed,
                 bar_width,
                 alpha=opacity,
                 color='y',
                 label='Proposed')

plt.xlabel('Source-target gender pair')
plt.ylabel('MCD (dB)')
plt.title('MCD by gender pair')
plt.ylim([0,8])
plt.xticks(index + bar_width, ('M-F','F-F','F-M','M-M'))
plt.legend()

plt.tight_layout()
plt.show()


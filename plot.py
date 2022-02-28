import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from utils import *
from upsetplot import UpSet
from upsetplot import from_contents
from statannot import add_stat_annotation
from scipy import stats
from scipy.spatial.distance import braycurtis
from skbio.stats.ordination import pcoa
from skbio.diversity.alpha import shannon as shannon_2
from skbio.diversity.alpha import simpson
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec


"""
Plot Figure 2
"""
font = {'size': 18}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(24, 12), constrained_layout=True)

# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gs = GridSpec(2, 4, figure=fig)
 
# create sub plots as grid
ax1 = fig.add_subplot(gs[:2, :2])

ax3 = fig.add_subplot(gs[1, 2:])
ax2 = fig.add_subplot(gs[0, 2:], sharex=ax3)

ax = ax1
palette = sns.color_palette('hls', n_colors=7, desat=0.99)
ax = sns.scatterplot(data=bt, x="Axis.1", y="Axis.2", hue="Study", edgecolor='black', style="Disease", palette=palette, ax=ax, s=80)
ax.set_xlabel("PCo1(28.3%)")
ax.set_ylabel('PCo2(14.2%)')
handles, labels = ax.get_legend_handles_labels()

extra1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
extra2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

legend1 = ax.legend(handles[:8]+[extra1, extra1], labels[:8] + [r"$R^2=%.3f$"%0.1641045, "$p=%.3f$"%0.000999001], loc='upper left', frameon=False, fontsize=13)
legend2 = ax.legend(handles[8:]+[extra1, extra1], labels[8:] + [r"$R^2=%.3f$"%0.01132199, "$p=%.3f$"%0.000999001], loc='upper left', frameon=False, fontsize=13, bbox_to_anchor=(0.2, 1))
ax.add_artist(legend1)
ax.set_title("(A)", x=-0.03, y=1)

# plot figure B
ax = ax2
palette = sns.color_palette('husl', n_colors=2)[::-1]
sns.boxplot(x='Study', y="Axis.1", hue="Disease", data=bt, ax=ax, palette=palette, linewidth=2.5, showfliers=False)
sns.stripplot(x='Study', y="Axis.1", hue="Disease", data=bt, jitter=True, dodge=True, marker='o', alpha=0.8, size=5,palette=palette, ax=ax)
add_stat_annotation(ax, data=bt, x='Study', y="Axis.1", hue="Disease",
                    box_pairs=[((dn, 'CRC'), (dn, 'Control')) for dn in official_name_mapping.values()],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=0, comparisons_correction=None)
# get legend information from the plot object
handles, labels = ax.get_legend_handles_labels()
# specify just one legend
ax.legend(handles[0:2], labels[0:2], loc='lower right', fancybox=True, fontsize=12)
ax.get_xaxis().set_visible(False)
ax.yaxis.tick_right()
ax.set_ylabel("PCo1")
ax.yaxis.set_label_position("right")
ax.set_title("(B)", x=0.02, y=1)

# plot figure C
ax = ax3
palette = sns.color_palette('husl', n_colors=2)[::-1]
sns.boxplot(x='Study', y="Axis.2", hue="Disease", data=bt, ax=ax, palette=palette, linewidth=2.5, showfliers=False)
sns.stripplot(x='Study', y="Axis.2", hue="Disease", data=bt, jitter=True, dodge=True, marker='o', alpha=0.8, size=5,palette=palette, ax=ax)
add_stat_annotation(ax, data=bt, x='Study', y="Axis.2", hue="Disease",
                    box_pairs=[((dn, 'CRC'), (dn, 'Control')) for dn in official_name_mapping.values()],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=0, comparisons_correction=None)
# get legend information from the plot object
ax.set_xlabel("")
handles, labels = ax.get_legend_handles_labels()
# specify just one legend
ax.legend(handles[0:2], labels[0:2], loc='lower right', fancybox=True, fontsize=12)
# ax.legend(loc='lower left')
ax.yaxis.tick_right()
ax.set_ylabel("PCo2")
ax.yaxis.set_label_position("right")
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.set_title("(C)", x=0.02, y=1)


"""
Plot Figure 3 (C) (D)
"""
fig, axs = plt.subplots(2, 2, figsize=(16,12), gridspec_kw={"height_ratios": [11,7], "width_ratios":[449,462]})
plt.subplots_adjust(wspace =0.02, hspace = 0.05)

for ax_row in axs:
    for ax in ax_row:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

cm = sns.color_palette("Spectral", as_cmap=True)

ax = axs[0][0]
sns.heatmap(np.log10(total_control_t_species.values), ax=ax,cbar=False, cmap=cm)
# ax.set_title('Control')
ax.yaxis.set_visible(True)
ax.axhline(y=0, color='k',linewidth=2)
ax.axhline(y=total_control_t_species.shape[0], color='k',linewidth=2)
ax.axvline(x=0, color='k',linewidth=2)
ax.axvline(x=total_control_t_species.shape[1], color='k',linewidth=2)

ax.set_title('(C)', loc='left', x=-1, y=1)
ax.xaxis.set_label_position('top')
ax.set_xlabel('Control', fontdict={'fontsize':20}) 
ax.xaxis.set_visible(True)
ax.set_xticks([])

ax = axs[0][1]

sns.heatmap(np.log10(total_case_t_species.values), ax=ax, cbar=False, cmap=cm)
# ax.set_title('CRC')
ax.axhline(y=0, color='k',linewidth=2)
ax.axhline(y=total_case_t_species.shape[0], color='k',linewidth=2)
ax.axvline(x=0, color='k',linewidth=2)
ax.axvline(x=total_case_t_species.shape[1], color='k',linewidth=2)

ax.xaxis.set_label_position('top')
ax.set_xlabel('CRC', fontdict={'fontsize':20}) 
ax.xaxis.set_visible(True)
ax.set_xticks([])

ax = axs[1][0]
sns.heatmap(np.log10(total_control_t_pwy), ax=ax,cbar=False, cmap=cm)
ax.yaxis.set_visible(True)
ax.axhline(y=0, color='k',linewidth=2)
ax.axhline(y=total_control_t_pwy.shape[0], color='k',linewidth=2)
ax.axvline(x=0, color='k',linewidth=2)
ax.axvline(x=total_control_t_pwy.shape[1], color='k',linewidth=2)
ax.xaxis.set_visible(True)
ax.set_xticks([])
ax.set_title('(D)', loc='left', x=-1, y=1)

ax = axs[1][1]
sns.heatmap(np.log10(total_case_t_pwy.values.clip(max=50000)+2), vmin=-1, ax=ax, cbar=False, cmap=cm)
ax.axhline(y=0, color='k',linewidth=2)
ax.axhline(y=total_case_t_pwy.shape[0], color='k',linewidth=2)
ax.axvline(x=0, color='k',linewidth=2)
ax.axvline(x=total_case_t_pwy.shape[1], color='k',linewidth=2)
cbaxes = fig.add_axes([0.91, 0.38, 0.018, 0.5]) 
fig.colorbar(ax.get_children()[0], cax=cbaxes, extend='min', label="log(abundance)")
plt.tight_layout()



"""
Plot Figure 5
"""
font = {'size': 18}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(24, 16), constrained_layout=True)

# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gs = GridSpec(2, 3, figure=fig)
 
# create sub plots as grid
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])

ax = ax1
classification_heatmap(official_name_mapping.values(), genome_rf_d, ax, level="Genome-level")
ax.set_title('(A)', loc='left')

ax = ax2
classification_heatmap(official_name_mapping.values(), species_rf_d, ax, level="Species-level")
# ax.get_yaxis().set_visible(False)
ax.set_title('(B)', loc='left')
ax.set_yticks([])

ax = ax3
classification_heatmap(official_name_mapping.values(), gf_rf_d, ax, level="Gene-family", cbar=True)
# ax.get_yaxis().set_visible(False)
ax.set_title('(C)', loc='left')
ax.set_yticks([])

ax = ax4
# palette = sns.color_palette('Set2', n_colors=3, desat=0.99)
palette = ('#e7e7e7', '#adadad', '#4c4c4c')
# palette = sns.color_palette()
sns.barplot(x='Study', y='AUROC', hue='Type of abundance', data=t, ax=ax, capsize=0.05, ci='sd', palette=palette, edgecolor='black', linewidth=2)
ax.set_title('(D)', loc='left')
ax.set_xlabel("")
ax.legend(loc='upper center', frameon=False, ncol=3)


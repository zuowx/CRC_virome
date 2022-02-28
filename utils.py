import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter
from upsetplot import UpSet
from upsetplot import from_contents
from statannot import add_stat_annotation
from scipy import stats
from scipy.spatial.distance import braycurtis
from skbio.stats.ordination import pcoa
from skbio.diversity.alpha import shannon as shannon_2
from skbio.diversity.alpha import simpson


def vote_bitscore(id_lineage_dict, match_table, genome_key=1, id_key=2, bit_key=12):
    tax_counter = {}
    for _, row in tqdm(match_table.iterrows()):
        genome_id = '_'.join(row[genome_key].split('_')[:-1])
        if genome_id not in tax_counter:
            tax_counter[genome_id] = {}
        if id_lineage_dict[row[id_key]] not in tax_counter[genome_id]:
            tax_counter[genome_id][id_lineage_dict[row[id_key]]] = [0, 0]
        tax_counter[genome_id][id_lineage_dict[row[id_key]]][0] += 1
        tax_counter[genome_id][id_lineage_dict[row[id_key]]][1] += row[bit_key]
    return tax_counter

def annotate(species_counter, level="s__"):
    
    def higher_counter():
        d = {}
        for k, v in species_counter.items():
            d[k] = {}
            for s, c in v.items():
                if s.find(level) != -1:
                    names = s.split('|')
                    for i in range(len(names)):
                        if names[i].find(level) != -1:
                            break
                    lineage = '|'.join(names[:i+1])
                    if lineage not in d[k]:
                        d[k][lineage] = [0,0]
                    d[k][lineage][0] += c[0]
                    d[k][lineage][1] += c[1]
        return d
    
    res = {}
    counter = species_counter if level == "s__" else higher_counter()
    for k, v in counter.items():
        if not isinstance(v, dict):
            print(v)
        l = sorted(v.items(), key=lambda x: x[-1], reverse=True)
        if sum([i[0] for i in v.values()]) <= 2 or (len(l) > 1 and l[0][1][0] == l[1][1][0] and l[0][1][1] == l[1][1][1]):
            continue
        res[k] = l[0][0]
    return res

def get_higher_level_dict(genome_tax_dict, higher_level="f__"):
    higher_dict = {}
    for k, v in genome_tax_dict.items():
        if v.find(higher_level) != -1:
            taxon_list = v.split('|')
            for i in range(len(taxon_list)):
                if taxon_list[i].startswith(higher_level):
                    break
            higher_dict[k] = '|'.join(taxon_list[:(i+1)])
        # else:
        #     higher_dict[k] = None
    return higher_dict

def get_tax_abundance_table(genome_table, genome_tax_dict, desc=None):
    tax_list = sorted(set(genome_tax_dict.values()))
    # tax_idx_d = {tax: i for i, tax in enumerate(tax_list)}
    # count_t =  genome_table.transpose()
    tax_genome_dict = {}
    for k, v in genome_tax_dict.items():
        if v in tax_genome_dict:
            tax_genome_dict[v].append(k)
        else:
            tax_genome_dict[v] = [k]
    new_table = pd.DataFrame(data=np.zeros([len(tax_list), genome_table.shape[1]]), index=tax_list, columns=genome_table.columns)
    genome_set = set(genome_table.index)
    for k, v in tqdm(tax_genome_dict.items(), desc=desc):
        for gid in v:
            if gid not in genome_set:
                continue
            new_table.loc[k,:] += genome_table.loc[gid,:]
    return new_table
    # for _, row in count_t.iterrows():
    #     new_count = [0 for _ in range(len(tax_list))]
    #     for 

def get_roc_acc_result(output_dir):
    d = {}
    dataset_names = ["zeller", "feng", "vogtmann", "thomas", "yu", "yang", "yamada"]
    for train_name in dataset_names:
        for test_name in dataset_names:
            name = train_name + '_' + test_name
            res = []
            with open(os.path.join(output_dir, name+'.txt'), 'r') as f:
                for line in f.readlines():
                    res.extend([float(i) for i in line.strip().split(',')])
            d[name] = res
    return d

def upset_plot(diff_dir, dataset_names, min_subset_size=0):
    d = {}
    for dn in dataset_names:
        t = pd.read_csv(os.path.join(diff_dir, dn+"_diff.csv"), index_col=0)
        t = t.loc[t.FDR < 0.05]
        d[dn] = list(set(t.index))
    upset_res = from_contents(d)
    ax = UpSet(upset_res, subset_size='count', min_subset_size=min_subset_size, sort_by='cardinality', orientation='vertical', show_counts='%d').plot()
    return upset_res, ax

def classification_heatmap(dataset_names, result_d, ax, cbar=False, level=""):
    roc_t = {k: [] for k in dataset_names}
    for train_name in dataset_names:
        for test_name in dataset_names:
            name = train_name+'&'+test_name
            roc_t[train_name].append(result_d[name][0])
            
    roc_t = pd.DataFrame.from_dict(roc_t, orient='index', columns=dataset_names)
    sns.heatmap(roc_t, linewidths=.5, annot=True, vmin=0.5, vmax=1.0, fmt=".3f", cmap="rocket_r", cbar=cbar, ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)
    ax.set_xlabel(level+" test set", fontsize=18)
    ax.set_ylabel(level+" training set", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)

def shannon(abundance_list):
    return shannon_2(abundance_list, base=np.e)

# convert abundance dict to abundance matrix
def get_xy(known_abundance_dict, label_dict):
    assert known_abundance_dict.keys() == label_dict.keys()
    X, y = [], []
    sample_id_list = list(label_dict.keys())
#     id_list = sorted(known_abundance_dict.keys())
    for sample_id in sample_id_list:
        # sample_id_list.append(sample_id)
        X.append(known_abundance_dict[sample_id])
        y.append(label_dict[sample_id])
    return np.array(X), np.array(y), sample_id_list

# calculate alpha diversity based on abundance dict
def dump_alpha_diversity_csv(tdict, ldict, age_dict=None, gender_dict=None, bmi_dict=None, path=None, normalize=False):
    assert tdict.keys() == ldict.keys()
    def normalize_vec(vec):
        total = sum(vec)
        if total == 0.0:
            total = 1.0
        return [value / total for value in vec]
    # sample_id_list = sorted(list(label_dict.keys()))
    res = []
    print("Calculate alpha diversity:")
    for name in tdict.keys():
        abd_t = tdict[name]
        lt = ldict[name]
        for ids in tqdm(abd_t.keys()):
            abd = abd_t[ids].to_list()
            if normalize:
                abd = normalize_vec(abd)
            item = [ids, shannon(abd), simpson(abd), 'CRC' if lt[ids].disease else "Control", name]
            if age_dict:
                item.append(age_dict[name][ids])
            if gender_dict:
                item.append(gender_dict[name][ids])
            if bmi_dict:
                item.append(bmi_dict[name][ids])
            res.append(item)
    col_names = ['SampleID', 'Shannon', 'Simpson', 'Phenotype', 'Study']
    if age_dict:
        col_names.append('Age')
    if gender_dict:
        col_names.append('Gender')
    if bmi_dict:
        col_names.append('BMI')
    t = pd.DataFrame(data=res, columns=col_names)
    
    if path:
        t.to_csv(path, index=False)
    return t

# calculate pairwise distance
def get_distance_matrix(abd_t, label_t, distance_func=braycurtis):
    n_sample = abd_t.shape[1]
    distance_matrix = np.zeros([n_sample, n_sample])
    abundance_dict = abd_t.to_dict(orient='list')
    label_dict = label_t.to_dict(orient='list')
    for k in label_dict.keys():
        label_dict[k] = label_dict[k][0]
    abundance_matrix, label_vec, sample_id_list = get_xy(abundance_dict, label_dict)
    for i in range(n_sample):
        for j in range(n_sample):
            distance = braycurtis(abundance_matrix[i, :], abundance_matrix[j, :])
            if np.isnan(distance):
                distance = 0
            distance_matrix[i, j] = distance
    return distance_matrix, abundance_matrix, label_vec, sample_id_list

# perform pcoa
def dump_pcoa_csv(tdict, ldict, age_dict=None, gender_dict=None, bmi_dict=None, dis_path=None, pcoa_path=None, normalize=False):
    def comman_features(tdict, normalize):
        s = next(iter(tdict.values())).index
        for v in tdict.values():
            s = s.union(v.index)
        print("# features: %d" % len(s))
        for k, v in tdict.items():
            diff_features = s.difference(v.index)
            tdict[k] = tdict[k].append(pd.DataFrame(data=np.zeros([len(diff_features), v.shape[1]]), index=diff_features, columns=v.columns))
            tdict[k] = tdict[k].loc[s]
        # res = {k: v.loc[s] for k, v in tdict.items()}
        res = tdict
        if normalize:
            for k, v in res.items():
                for ids in v.keys():
                    sumup = res[k][ids].sum()
                    if sumup == 0:
                        sumup = 1
                    res[k][ids] = res[k][ids] / sumup
        return res
    abd_t = pd.concat(comman_features(tdict, normalize), axis=1)
    label_t = pd.concat(ldict, axis=1)
    distance_matrix, abundance_matrix, label_vec, sample_id_list = get_distance_matrix(abd_t, label_t)
    if dis_path:
        # pairwise_names = ['.'.join(i) for i in sample_id_list]
        pairwise_names = [i[1] for i in sample_id_list]
        pd.DataFrame(distance_matrix, columns=pairwise_names, index=pairwise_names).to_csv(dis_path)
    print("PCoA...")
    pcoa_result = pcoa(distance_matrix, number_of_dimensions=2)
    pcoa_coord = pcoa_result.samples.values
    prop_exp = pcoa_result.proportion_explained
    t = []
    for i in range(len(sample_id_list)):
        item = [sample_id_list[i][1], sample_id_list[i][0], pcoa_coord[i, 0], pcoa_coord[i, 1], 'CRC' if label_vec[i] else 'Control']
        ids = sample_id_list[i][1]
        name = sample_id_list[i][0]
        if age_dict:
            item.append(age_dict[name][ids])
        if gender_dict:
            item.append(gender_dict[name][ids])
        if bmi_dict:
            item.append(bmi_dict[name][ids])
        t.append(item)
    
    col_names = ["SampleID", "Study", "Axis.1", "Axis.2", "Phenotype"]
    if age_dict:
        col_names.append('Age')
    if gender_dict:
        col_names.append('Gender')
    if bmi_dict:
        col_names.append('BMI')
    t = pd.DataFrame(data=t, columns=col_names)
    if pcoa_path:
        t.to_csv(pcoa_path, index=False)
    return t, prop_exp

def alpha_diversity_boxplot(ax, t, type_key="Phenotype", div_key="Shannon", disease=("Control", "CRC"), pval=False, rotate_x=True):
    palette = sns.color_palette('husl', n_colors=2)[::-1]
    sns.boxplot(x='Study', y=div_key, hue=type_key, data=t, ax=ax, palette=palette, linewidth=2.5, showfliers=False)
    sns.stripplot(x='Study', y=div_key, hue=type_key, data=t, jitter=True, dodge=True, marker='o', alpha=0.8, size=5,palette=palette, ax=ax)
    # get legend information from the plot object
    handles, labels = ax.get_legend_handles_labels()
    # specify just one legend
    ax.legend(handles[0:2], labels[0:2], loc='lower left', fancybox=True)
    # ax.legend(loc='lower left')
    if rotate_x:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    return ax
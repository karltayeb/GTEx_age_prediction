import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

training_data = pickle.load(open('GTEx_train'))
test_data = pickle.load(open('GTEx_test'))
tissues = training_data.keys()


file_paths = [tissue + '_tuning' for tissue in tissues]
tuning = pd.concat([pd.read_pickle(file_path).T.reset_index().rename(
    columns={'level_0': 'regressed_pcs', 'level_1': 'penalty'})
    for file_path in file_paths], keys=tissues)
tuning = tuning.reset_index().rename(
    columns={'level_0': 'tissue'}).drop('level_1', axis=1)
tuning = pd.melt(
    tuning,
    id_vars=['tissue', 'regressed_pcs', 'penalty'],
    value_name='score').drop('variable', axis=1)


i = 0
fig, ax = plt.subplots(4, sharex=True, sharey=True, figsize=(5, 12))
for tissue, group in tuning.groupby('tissue'):
    print tissue
    alpha = group.set_index(
        ['regressed_pcs']).groupby('penalty').score.mean().argmax()
    sns.boxplot(
        data=group[group.penalty == alpha],
        y='regressed_pcs', x='score', ax=ax[i], orient='h')
    ax[i].set_title(tissue)
    i += 1

plt.savefig('scores_pc_by_tissue')
plt.close()


test_results = pd.read_pickle('test_results')
X = test_results[test_results.tissue == 'muscle_skeletal']
X = X[X.test_set_size == 'full']

sns.boxplot(data=X, x='training_set_size', y='value', hue='type')
plt.ylabel('score')
plt.title('Muscle-Skeletal test set scores')
plt.savefig('fig1')
plt.close()


X = test_results[test_results.test_set_size == 'full']
g = sns.factorplot(
    data=X, x='training_set_size', y='value', col='tissue', hue='type')
plt.savefig('fig2')
plt.close()

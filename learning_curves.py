import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

from collections import defaultdict

training_data = pickle.load(open('./data/GTEx_train'))
test_data = pickle.load(open('./data/GTEx_test'))
tissues = training_data.keys()

scores = defaultdict(list)
permuted_scores = defaultdict(list)
min_scores = defaultdict(list)
min_permuted_scores = defaultdict(list)

for tissue in tissues:
    file_path = tissue + '_tuning'
    print(file_path)
    tuning = pd.read_pickle(file_path)
    k, alpha = tuning.max().argmax()

    gene_expression, phenotype = training_data[tissue]
    gene_expression_test, phenotype_test = test_data[tissue]
    gene_expression = gene_expression.T
    gene_expression_test = gene_expression_test.T

    total_samples = gene_expression.shape[0]

    for subset in np.arange(25, total_samples, 25):
        for runs in range(20):
            # select a random percent of of training set
            selected_samples = np.random.choice(
                total_samples, subset, replace=False)
            X = gene_expression.iloc[selected_samples]
            subset_phenotype = phenotype.loc[X.index]

            X_test = gene_expression_test

            gender = \
                (subset_phenotype.GENDER - 1).as_matrix().reshape(-1, 1)
            gender_test = \
                (phenotype_test.GENDER - 1).as_matrix().reshape(-1, 1)

            y = subset_phenotype.AGE.as_matrix()
            y_perm = y[np.random.permutation(y.size)]
            y_test = phenotype_test.AGE.as_matrix()

            # scale features, do PCA
            scaler1 = StandardScaler().fit(X)
            X = scaler1.transform(X)
            X_test = scaler1.transform(X_test)

            if k > 0:
                pca = PCA(n_components=k).fit(X)
                pcs = pca.transform(X)
                pcs_test = pca.transform(X_test)

                # correct for age and top k pcs
                regressors = np.hstack([pcs, gender])
                regressors_test = np.hstack([pcs_test, gender_test])
            else:
                regressors = gender
                regressors_test = gender_test

            lm = LinearRegression().fit(regressors, X)
            X_corrected = X - lm.predict(regressors)
            X_corrected_test = X_test - lm.predict(regressors_test)

            scaler2 = StandardScaler().fit(X_corrected)
            X_final = scaler2.transform(X_corrected)
            X_final_test = scaler2.transform(X_corrected_test)

            lasso = Lasso(alpha=alpha).fit(X_final, y)
            lasso_perm = Lasso(alpha=alpha).fit(X_final, y_perm)

            scores[(tissue, subset, 'real', 'full')].append(
                lasso.score(X_final_test, y_test))
            permuted_scores[(tissue, subset, 'permuted', 'full')].append(
                lasso_perm.score(X_final_test, y_test))

            selected_test_samples = np.random.choice(
                X_final_test.shape[0], 70, replace=False)
            X_final_test_sub = X_final_test[selected_test_samples, :]
            y_test_sub = y_test[selected_test_samples]

            min_scores[(tissue, subset, 'real', 'min')].append(
                lasso.score(X_final_test_sub, y_test_sub))
            min_permuted_scores[(tissue, subset, 'permuted', 'min')].append(
                lasso_perm.score(X_final_test_sub, y_test_sub))


real = pd.DataFrame.from_dict(scores)
real = real.T.reset_index()

permuted = pd.DataFrame.from_dict(permuted_scores)
permuted = permuted.T.reset_index()

min_real = pd.DataFrame.from_dict(min_scores)
min_real = min_real.T.reset_index()

min_permuted = pd.DataFrame.from_dict(min_permuted_scores)
min_permuted = min_permuted.T.reset_index()

everything = pd.concat([real, permuted, min_real, min_permuted])
everything = everything.rename(
    columns={'level_0': 'tissue',
             'level_1': 'training_set_size',
             'level_2': 'type',
             'level_3': 'test_set_size'}
    )

everything = pd.melt(
    everything,
    id_vars=['tissue', 'training_set_size', 'type', 'test_set_size'],
    value_vars=[0, 1, 2]
    )

pickle.dump(everything, 'test_results')

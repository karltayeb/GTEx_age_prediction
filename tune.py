import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

import pickle

import sys

if __name__ == "__main__":

    gene_expression_path = sys.argv[1]
    phenotype_path = sys.argv[2]
    tissue_name = sys.argv[3]

    gene_expression = pd.read_csv(gene_expression_path, sep='\t')
    phenotype = pd.read_csv(phenotype_path, sep='\t')
    phenotype.AGE = phenotype.AGE.apply(lambda age: int(age[:2]))

    # split data into training and test sets
    np.random.seed(0)
    gene_expression = gene_expression.drop(
        ['#chr', 'start', 'end'], axis=1).set_index('gene_id')
    msk = np.random.rand(len(gene_expression.columns)) < 0.7
    gene_expression_train = gene_expression.loc[:, msk].T
    gene_expression_test = gene_expression.loc[:, ~msk].T
    phenotype_train = phenotype.set_index(
        'SUBJID').loc[gene_expression_train.index]
    phenotype_test = phenotype.set_index(
        'SUBJID').loc[gene_expression_test.index]

    total_samples = gene_expression_train.shape[0]
    lassos = {}

    for k in range(2):
        for alpha in np.array([10 * (0.5**x) for x in range(2)]):
            lassos[(k, alpha)] = []
            for train, val in KFold(total_samples, 2):
                X = gene_expression_train.iloc[train].as_matrix()
                X_val = gene_expression_train.iloc[val].as_matrix()

                gender = (phenotype_train.GENDER.iloc[train] - 1
                          ).as_matrix().reshape(-1, 1)
                gender_val = (phenotype_train.GENDER.iloc[val] - 1
                              ).as_matrix().reshape(-1, 1)

                y = (phenotype_train.AGE.iloc[train]).as_matrix()
                y_val = (phenotype_train.AGE.iloc[val]).as_matrix()

                # scale features, do PCA
                scaler1 = StandardScaler().fit(X)
                X = scaler1.transform(X)
                X_val = scaler1.transform(X_val)

                pca = PCA().fit(X)
                pcs = pca.transform(X)
                pcs_val = pca.transform(X_val)

                # correct for age and top k pcs
                regressors = np.hstack([pcs[:, :k], gender])
                regressors_val = np.hstack([pcs_val[:, :k], gender_val])

                lm = LinearRegression().fit(regressors, X)
                X_corrected = X - lm.predict(regressors)
                X_corrected_val = X_val - lm.predict(regressors_val)

                scaler2 = StandardScaler().fit(X_corrected)
                X_final = scaler2.transform(X_corrected)
                X_final_val = scaler2.transform(X_corrected_val)

                lasso = Lasso(alpha=alpha).fit(X_final, y)
                lasso.score(X_final_val, y_val)
                lassos[(k, alpha)].append(lasso.score(X_final_val, y_val))

    pickle_path = tissue_name + '_tuning'
    pickle.dump(lassos, open(pickle_path, 'wb'))

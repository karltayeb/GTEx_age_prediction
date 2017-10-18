import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

import pickle

import sys

if __name__ == "__main__":

    tissue_index = int(sys.argv[1])

    training_data = pickle.load(open('GTEx_train'))
    test_data = pickle.load(open('GTEx_test'))
    tissues = training_data.keys()

    tissue = tissues[tissue_index]
    gene_expression_train, phenotype_train = training_data[tissue]
    gene_expression_train = gene_expression_train.T

    total_samples = gene_expression_train.shape[0]
    lassos = {}

    KRANGE = int(sys.argv[2])
    KFOLDS = int(sys.argv[3])

    print("Tuning penalty for", tissue)
    for k in range(KRANGE):
        for alpha in np.array([10 * (0.5**x) for x in range(2)]):
            lassos[(k, alpha)] = []
            for train, val in KFold(total_samples, KFOLDS):
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

    lassos = pd.DataFrame.from_dict(lassos)
    pickle_path = tissue + '_tuning'
    pickle.dump(lassos, open(pickle_path, 'wb'))

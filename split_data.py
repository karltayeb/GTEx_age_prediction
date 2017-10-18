import pandas as pd
import numpy as np
import pickle
import sys

if __name__ == "__main__":
    matrix_dir = sys.argv[1]
    phenotype_dir = sys.argv[2]
    out_dir = sys.argv[3]

    muscle_skeletal_path = matrix_dir + '/Muscle_Skeletal_Analysis.v6p.normalized.expression.bed.gz'
    whole_blood_path = matrix_dir + '/Whole_Blood_Analysis.v6p.normalized.expression.bed.gz'
    adipose_subcutaneous_path = matrix_dir + '/Adipose_Subcutaneous_Analysis.v6p.normalized.expression.bed.gz'
    thyroid_path = matrix_dir + '/Thyroid_Analysis.v6p.normalized.expression.bed.gz'
    phenotype_path = phenotype_dir + '/GTEx_Data_V6_Annotations_SubjectPhenotypesDS.txt'

    print(muscle_skeletal_path)
    muscle_skeletal_data = pd.read_csv(muscle_skeletal_path, sep='\t', compression='gzip')
    print(whole_blood_path)
    whole_blood_data = pd.read_csv(whole_blood_path, sep='\t', compression='gzip')
    print(adipose_subcutaneous_path)
    adipose_subcutaneous_data = pd.read_csv(adipose_subcutaneous_path, sep='\t', compression='gzip')
    print(thyroid_path)
    thyroid_data = pd.read_csv(thyroid_path, sep='\t', compression='gzip')

    phenotype = pd.read_csv(phenotype_path, sep='\t')
    phenotype.AGE = phenotype.AGE.apply(lambda age: int(age[:2]))

    # 70/30 split of training/test data for each tissue

    muscle_skeletal_data = muscle_skeletal_data.drop(['#chr', 'start', 'end'],axis=1).set_index('gene_id')
    msk = np.random.rand(len(muscle_skeletal_data.columns)) < 0.7
    muscle_skeletal_train = muscle_skeletal_data.loc[:, msk]
    muscle_skeletal_test = muscle_skeletal_data.loc[:, ~msk]
    muscle_skeletal_phenotype_train = phenotype.set_index('SUBJID').loc[muscle_skeletal_train.columns]
    muscle_skeletal_phenotype_test = phenotype.set_index('SUBJID').loc[muscle_skeletal_test.columns]


    whole_blood_data = whole_blood_data.drop(['#chr', 'start', 'end'],axis=1).set_index('gene_id')
    msk = np.random.rand(len(whole_blood_data.columns)) < 0.7
    whole_blood_train = whole_blood_data.loc[:, msk]
    whole_blood_test = whole_blood_data.loc[:, ~msk]
    whole_blood_phenotype_train = phenotype.set_index('SUBJID').loc[whole_blood_train.columns]
    whole_blood_phenotype_test = phenotype.set_index('SUBJID').loc[whole_blood_test.columns]

    adipose_subcutaneous_data = adipose_subcutaneous_data.drop(['#chr', 'start', 'end'],axis=1).set_index('gene_id')
    msk = np.random.rand(len(adipose_subcutaneous_data.columns)) < 0.7
    adipose_subcutaneous_train = adipose_subcutaneous_data.loc[:, msk]
    adipose_subcutaneous_test = adipose_subcutaneous_data.loc[:, ~msk]
    adipose_subcutaneous_phenotype_train = phenotype.set_index('SUBJID').loc[adipose_subcutaneous_train.columns]
    adipose_subcutaneous_phenotype_test = phenotype.set_index('SUBJID').loc[adipose_subcutaneous_test.columns]

    thyroid_data = thyroid_data.drop(['#chr', 'start', 'end'],axis=1).set_index('gene_id')
    msk = np.random.rand(len(thyroid_data.columns)) < 0.7
    thyroid_train = thyroid_data.loc[:, msk]
    thyroid_test = thyroid_data.loc[:, ~msk]
    thyroid_phenotype_train = phenotype.set_index('SUBJID').loc[thyroid_train.columns]
    thyroid_phenotype_test = phenotype.set_index('SUBJID').loc[thyroid_test.columns]

    train = {
        'muscle_skeletal': (muscle_skeletal_train, muscle_skeletal_phenotype_train),
        'whole_blood': (whole_blood_train, whole_blood_phenotype_train),
        'adipose_subcutaneous': (adipose_subcutaneous_train, adipose_subcutaneous_phenotype_train),
        'thyroid': (thyroid_train, thyroid_phenotype_train)
    }

    test = {
        'muscle_skeletal': (muscle_skeletal_test, muscle_skeletal_phenotype_test),
        'whole_blood': (whole_blood_test, whole_blood_phenotype_test),
        'adipose_subcutaneous': (adipose_subcutaneous_test, adipose_subcutaneous_phenotype_test),
        'thyroid': (thyroid_test, thyroid_phenotype_test)
    }

    pickle.dump(train, open(out_dir + '/GTEx_train', 'wb'))
    pickle.dump(test, open(out_dir + '/GTEx_test', 'wb'))
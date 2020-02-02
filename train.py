import numpy as np 
import pandas as pd 
import os

train = pd.read_csv('trainpro.csv')
test = pd.read_csv('testpro.csv')


lab = train['label']
print('ok')
f = train.drop(columns=['id','label'])
testf = test.drop(columns=['id'])

from sklearn.decomposition import PCA, FastICA

pca = PCA(n_components=12, random_state=420)
pca1_results_train = pca.fit_transform(f)
pca1_results_test = pca.transform(testf)

ica = FastICA(n_components=12, random_state=420)
ica1_results_train = ica.fit_transform(f)
ica1_results_test = ica.transform(testf)

trainNN = pd.DataFrame()
testNN = pd.DataFrame()


for i in range(1, 21):
    train['pca_' + str(i)] = pca1_results_train[:,i-1]
    test['pca_' + str(i)] = pca1_results_test[:, i-1]

    train['ica_' + str(i)] = ica1_results_train[:,i-1]
    test['ica_' + str(i)] = ica1_results_test[:, i-1]

from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=25, random_state=420)
tsvd_results_train = tsvd.fit_transform(f)
tsvd_results_test = tsvd.transform(testf)


from sklearn.random_projection import GaussianRandomProjection
grp = GaussianRandomProjection(n_components=25, eps=0.2, random_state=420)
grp_results_train = grp.fit_transform(f)
grp_results_test = grp.transform(testf)


from sklearn.random_projection import SparseRandomProjection
srp = SparseRandomProjection(n_components=25, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(f)
srp_results_test = srp.transform(testf)

for i in range(1, 26):
    trainNN['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    testNN['tsvd_' + str(i)] = tsvd_results_test[:, i-1]

    trainNN['grp_' + str(i)] = grp_results_train[:,i-1]
    testNN['grp_' + str(i)] = grp_results_test[:, i-1]

    trainNN['srp_' + str(i)] = srp_results_train[:,i-1]
    testNN['srp_' + str(i)] = srp_results_test[:, i-1]

import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.98,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(f, target)
dtest = xgb.DMatrix(testf)

num_boost_rounds = 300

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)


#trainNN.shape
sub = pd.DataFrame()

#sub['ID_code'] = test['ID_code']
sub['target'] = y_pred
sub.to_csv('stacked-models.csv', index=False)
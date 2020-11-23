import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style='white')
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.cluster import MiniBatchKMeans
import ml_metrics as metrics 
from sklearn.ensemble import RandomForestClassifier
from utils import *

input_path = '../data/' # original dataset
output_path = '../mod_data/' # dataset after feature engineering
result_path = '../results/'

print('load data')
train = pd.read_csv(output_path+'train.csv', nrows=10000)

print('Normalize Data')
x = train.iloc[:, :-1].values 
y = train.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print('train KNN')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)


train_preds = knn.predict_proba(x_train)
test_preds = knn.predict_proba(x_test)
train_index = top_k_labels(train_preds, 5)
test_index = top_k_labels(test_preds, 5)
train_labels = [[l.tolist()] for l in y_train]
test_labels = [[l.tolist()] for l in y_test]

knn_train_score = metrics.mapk(train_labels, train_index.tolist(), 5)
knn_test_score = metrics.mapk(test_labels, test_index.tolist(), 5)
print('knn_train, %s, knn_test %s'%(knn_train_score, knn_test_score))

print('Train Randon Forest')
rf = RandomForestClassifier(max_depth=10)
rf.fit(x_train, y_train)

train_preds = rf.predict_proba(x_train)
test_preds = rf.predict_proba(x_test)
train_index = top_k_labels(train_preds, 5)
test_index = top_k_labels(test_preds, 5)
train_labels = [[l.tolist()] for l in y_train]
test_labels = [[l.tolist()] for l in y_test]

rf_train_score = metrics.mapk(train_labels, train_index.tolist(), 5)
rf_test_score = metrics.mapk(test_labels, test_index.tolist(), 5)
print('rf_train, %s, rf_test %s'%(rf_train_score, rf_test_score))


from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim 

class Dataset_py(Dataset):
	def __init__(self, x, y):
		self.x = torch.tensor(x, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.long)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


y_train_array = y_train.reshape((len(y_train), 1))
y_test_array = y_test.reshape((len(y_test), 1))

batch_size = 64

train_ds = Dataset_py(x_train, y_train)

train_dl = DataLoader(train_ds, batch_size=batch_size)

xtrain_t = torch.tensor(x_train, dtype=torch.float32)

xtest_t = torch.tensor(x_test, dtype=torch.float32)

label_num = np.max(train['hotel_cluster'].values)+1


class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.net = nn.Sequential(
		nn.Linear(input_size, 128), 
		nn.ReLU(), 
		nn.Linear(128, 64), 
		nn.ReLU(),
		nn.Linear(64, output_size),
		# nn.Softmax(),
		)

	def forward(self, x):
		pred = self.net(x)
		return pred 


nn_model = MLP(x_train.shape[1], label_num)
cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
loss_list = []

epoch_num = 200
for epoch in range(epoch_num):
  for x_batch, y_batch in train_dl:
      pred = nn_model(x_batch)
      loss = cost(pred, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  print('epoch %s, error %s'%(epoch, loss.item()))
  loss_list.append(loss.item())

# plt.figure(figsize=(6,4))
# plt.plot(loss_list)
# plt.xlabel('Epcoh num', fontsize=12)
# plt.ylabel('Loss',fontsize=12)
# plt.tight_layout()
# # plt.savefig(result_path+'nn_learning_curve'+'.png', dpi=100)
# plt.show()

nn_model.eval()
train_preds = torch.log_softmax(nn_model(xtrain_t), dim=1).detach().numpy()
test_preds = torch.log_softmax(nn_model(xtest_t), dim=1).detach().numpy()
# print('train_preds', train_preds)
train_index = top_k_labels(train_preds, 5)
test_index = top_k_labels(test_preds, 5)
train_labels = [[l.tolist()] for l in y_train]
test_labels = [[l.tolist()] for l in y_test]

nn_train_score = metrics.mapk(train_labels, train_index.tolist(), 5)
nn_test_score = metrics.mapk(test_labels, test_index.tolist(), 5)
print('nn_train, %s, nn_test %s'%(nn_train_score, nn_test_score))

import xgboost as xgb
x = train.iloc[: :-1].values 
y = train.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
dvalid = xgb.DMatrix(x_val, label=y_val)


watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

params = {}
params['objective'] = 'multi:softprob'
params['eval_metric'] = 'mlogloss'
params['num_class'] = 100
params['max_depth'] = 20
params['eta'] = 0.1
params['min_child_weight'] = 20


print('Training XGBoost')
model = xgb.train(params, dtrain, 200, watchlist, early_stopping_rounds=50, verbose_eval=True)


train_preds = model.predict(dtrain)
test_preds = model.predict(dtest)
train_index = top_k_labels(train_preds, 5)
test_index = top_k_labels(test_preds, 5)
train_labels = [[l.tolist()] for l in y_train]
test_labels = [[l.tolist()] for l in y_test]
xgb_train_score = metrics.mapk(train_labels, train_index.tolist(), 5)
xgb_test_score = metrics.mapk(test_labels, test_index.tolist(), 5)
print('xgb_train, %s, xgb_test %s'%(xgb_train_score, xgb_test_score))

model_list = ['KNN', 'Random Forest', 'Neural Network', 'XGBoost']
train_error_list = [knn_train_score, rf_train_score, nn_train_score, xgb_train_score]
test_error_list = [knn_test_score, rf_test_score, nn_test_score, xgb_test_score]

plt.figure(figsize=(6,4))
plt.bar(np.arange(len(model_list)), train_error_list, width=0.2,  color='lightblue', align='center', label='Train')
plt.bar(np.arange(len(model_list))-0.2, test_error_list, width=0.2, color='y', align='center', label='Test')
plt.xticks(np.arange(len(model_list)), model_list, rotation=45)
plt.ylim([0, 1])
plt.xlabel('Models', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'find_tune_models_error'+'.png', dpi=100)
plt.show()

print('train_error_list', train_error_list)
print('test_error_list', test_error_list)

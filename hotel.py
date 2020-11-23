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
input_path = '../data/' # original dataset
output_path = '../mod_data/' # dataset after feature engineering
result_path = '../results/'

# N = 500000
# des = pd.read_csv(input_path+'destinations.csv', nrows=N)
# train = pd.read_csv(input_path+'train.csv', nrows=N)
# # test = pd.read_csv(input_path+'test.csv', nrows=N)
# train = train.drop(['is_booking', 'cnt'], axis=1)

# # train.head(2)
# # train.info()
# # train.dtypes
# # train.count()


# total = train.isnull().sum().sort_values(ascending=False)
# percent = (train.isnull().sum()/train['hotel_cluster'].count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# train['date_time'] = pd.to_datetime(train['date_time'])
# train['srch_co'] = pd.to_datetime(train['srch_co'])
# train['srch_ci'] = pd.to_datetime(train['srch_ci'])
# train['night_num'] = (train['srch_co'] - train['srch_ci']).astype('timedelta64[D]')
# train['adv_days'] = (train['srch_ci'] - train['date_time']).astype('timedelta64[D]')
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# cal = calendar()
# holidays = cal.holidays(start=train.srch_ci.min(), end=train.srch_ci.max())
# train['holiday'] = train['srch_ci'].isin(holidays)
# train['ci_day'] = train['srch_ci'].dt.day
# train['ci_month'] = train['srch_ci'].dt.month
# train['co_day'] = train['srch_co'].dt.day
# train['co_month'] = train['srch_co'].dt.month
# train['srch_day'] = train['date_time'].dt.day
# train['srch_month'] = train['date_time'].dt.month
# train['srch_year'] = train['date_time'].dt.year
# # fill na 
# train['night_num'] = train['night_num'].fillna(0)
# train['ci_day'] = train['ci_day'].fillna(0)
# train['ci_month'] = train['ci_month'].fillna(0)
# train['co_day'] = train['co_day'].fillna(0)
# train['co_month'] = train['co_month'].fillna(0)
# train['adv_days'] = train['adv_days'].fillna(0)
# train['orig_destination_distance'].fillna(train['orig_destination_distance'].mean(), inplace = True)
# train['holiday'] = 1.0*(train.holiday.values == True)

# train = train.drop(['srch_ci', 'srch_co', 'date_time'], axis=1)
# columns = ['site_name', 'posa_continent', 'user_location_country',
#        'user_location_region', 'user_location_city',
#        'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
#        'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
#        'srch_destination_id', 'srch_destination_type_id', 'hotel_continent',
#        'hotel_country', 'hotel_market',  'night_num',
#        'adv_days', 'holiday', 'ci_day', 'ci_month', 'co_day', 'co_month',
#        'srch_day', 'srch_month', 'srch_year', 'hotel_cluster']
# train = train[columns]
# train.to_csv(output_path+'train.csv', index=False)

# columns = ['date_time', 'site_name', 'posa_continent', 'user_location_country',
#        'user_location_region', 'user_location_city',
#        'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
#        'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt',
#        'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id',
#        'hotel_continent', 'hotel_country', 'hotel_market', 'hotel_cluster',
#        'night_num', 'adv_days', 'holiday', 'ci_day', 'ci_month']


# box_columns = ['site_name', 'posa_continent', 'user_location_country',
#        'user_location_region', 'user_location_city',
#        'orig_destination_distance', 'user_id', 
#        'channel', 'srch_adults_cnt', 'srch_children_cnt',
#        'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id',
#        'hotel_continent', 'hotel_country', 'hotel_market', 'hotel_cluster','night_num', 'adv_days', 'holiday', 'ci_day', 'ci_month']

# for i in range(len(box_columns)):
# 	plt.figure(figsize=(6,4))
# 	sns.boxplot(x=train[box_columns[i]])
# 	plt.tight_layout()
# 	plt.savefig(result_path+'boxplot'+box_columns[i]+'.png', dpi=100)
# 	plt.show()

# hist_columns = ['is_mobile', 'is_package','hotel_continent', 'hotel_country', 'hotel_market', 'hotel_cluster', 'user_location_country',
#        'user_location_region', 'user_location_city',
#        'orig_destination_distance','srch_adults_cnt', 'srch_children_cnt',
#        'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id' ]
# for i in range(len(hist_columns)):
# 	plt.figure(figsize=(6,4))
# 	sns.histplot(x=train[num_columns[i]])
# 	plt.tight_layout()
# 	plt.savefig(result_path+'hist'+hist_columns[i]+'.png', dpi=100)
# 	plt.show()


# sns.countplot(x='hotel_continent', data=train)
# plt.show()
# sns.countplot(x='hotel_continent', hue='posa_continent', data=train)
# plt.show()

# plt.figure(figsize=(6,4))
# sns.distplot(train['user_location_country'], label="User country")
# sns.distplot(train['hotel_country'], label="Hotel country")
# plt.legend()
# plt.tight_layout()
# plt.savefig(result_path+'country_dist'+'.png', dpi=100)
# plt.show()

train = pd.read_csv(output_path+'train.csv')

x = train.iloc[:, :-1].values 
y = train.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import ml_metrics as metrics 
from utils import *

rf = RandomForestClassifier(max_depth=10)
rf.fit(x_train, y_train)
rf_yt = rf.predict(x_train)
accu_t = accuracy_score(y_train, rf_yt)
# 0.16
cfm_t = confusion_matrix(y_train, rf_yt)

plt.matshow(cfm_t, cmap=plt.cm.Reds)
plt.savefig(result_path+'train_cfm'+'.png', dpi=100)
plt.show()

# row_sums = cfm_t.sum(axis=1, keepdims=True) 
# norm_cfm_t = cfm_t / row_sums
# np.fill_diagonal(norm_cfm_t, 0) 
# plt.matshow(norm_cfm_t, cmap=plt.cm.Reds) 
# plt.show()

rf_y = rf.predict(x_test)
accu = accuracy_score(y_test, rf_y)
# 0.14
cfm = confusion_matrix(y_test, rf_y)

# plt.matshow(cfm, cmap=plt.cm.Reds)
# plt.savefig(result_path+'test_cfm'+'.png', dpi=100)
# plt.show()

train_preds = rf.predict_proba(x_train)
test_preds = rf.predict_proba(x_test)
train_index = top_k_labels(train_preds, 5)
test_index = top_k_labels(test_preds, 5)

train_labels = [[l.tolist()] for l in y_train]
test_labels = [[l.tolist()] for l in y_test]

train_score = metrics.mapk(train_labels, train_index.tolist(), 5)
test_score = metrics.mapk(test_labels, test_index.tolist(), 5)
# 0.26, 0.23

user_feats = ['site_name', 'posa_continent', 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance', 'user_id', 'is_mobile', 'is_package']
hotel_feats = ['hotel_continent', 'hotel_country', 'hotel_market']
user_data = train[user_feats]
hotel_data = train[hotel_feats]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=100)
kmeans.fit(user_data)
train.loc[:,'user_cluster'] = kmeans.labels_
kmeans = KMeans(n_clusters=100)
kmeans.fit(hotel_data)
train.loc[:,'hotel_new_cluster'] = kmeans.labels_

columns = ['site_name', 'posa_continent', 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
       'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
       'srch_destination_id', 'srch_destination_type_id', 'hotel_continent',
       'hotel_country', 'hotel_market', 'night_num', 'adv_days', 'holiday',
       'ci_day', 'ci_month', 'co_day', 'co_month', 'srch_day', 'srch_month',
       'srch_year',  'user_cluster', 'hotel_new_cluster', 'hotel_cluster']

train = train[columns] 
# train.to_csv(output_path+'train.csv', index=False)

plt.figure(figsize=(6,4))
sns.distplot(train['user_cluster'])
plt.savefig(result_path+'user_cluster'+'.png', dpi=100)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.distplot(train['hotel_cluster'])
plt.savefig(result_path+'hotel_cluster'+'.png', dpi=100)
plt.tight_layout()
plt.show()


x = train.iloc[:, :-1].values 
y = train.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# rf = RandomForestClassifier(max_depth=10)
# rf.fit(x_train, y_train)

# imp = rf.feature_importances_
# indices = np.argsort(imp)[::-1]

# feats = train.columns[:-1][list(indices)]

# plt.figure(figsize=(6,4))
# plt.bar(range(x_train.shape[1]), imp[indices], color='lightblue', align='center')
# plt.xticks(range(x_train.shape[1]), feats, rotation=90)
# plt.xlim([-1, x_train.shape[1]])
# # plt.xlabel('features', fontsize=12)
# plt.ylabel('feature importance', fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'random_forest_feature_importance'+'.png', dpi=100)
# plt.show()


# cov_mat = np.cov(x_train.T)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# tot = sum(eigen_vals)
# var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)

# plt.figure(figsize=(6,4))
# plt.bar(range(x_train.shape[1]), var_exp, alpha=0.5, color='lightblue', align='center', label='individual explained variance')
# plt.step(range(x_train.shape[1]), cum_var_exp, where='mid', color='lightblue', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio', fontsize=12)
# plt.xlabel('Principle Components', fontsize=12)
# plt.legend(loc=0, fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'pca_score'+'.png', dpi=100)
# plt.show()





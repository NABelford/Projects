#Modeling County Data
# Target = DeathsPerCapita

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Reading in and examining data
data_all = pd.read_table('flatfile.csv', sep=',')
data_all.head()
data_all.columns
data_all.shape

#removing rows from puerto rico and unincorporated areas for which there is no data
data = data_all[data_all['FIPSCode'] < 60000]

data.head()
data.columns
data.shape

#printing number of missing values for each column
for col in data.columns.values :
    print(col, " NA: ", data[col].isnull().sum())


#Various plots to explore realtionships among variables
data['proportionVotesWon_Other'].plot.hist()

data.plot.scatter(x='medianHouseholdIncome', y='medianHousingPrice')
plt.show()
data.plot.scatter(x='medianHouseholdIncome', y='DeathsPerCapita')
plt.show()
data.plot.scatter(x='medianHousingPrice', y='DeathsPerCapita')
plt.show()
data.plot.scatter(x='medianHouseholdIncome', y='CasesPerCapita')
plt.show()
data.plot.scatter(x='medianHousingPrice', y='CasesPerCapita')
plt.show()

data_mod.plot.scatter(x='proportionVotesWon_Other', y='DeathsPerCapita', alpha=0.4)
data_mod.plot.scatter(x='medianHousingPrice', y='DeathsPerCapita', alpha=0.4)
data_mod.plot.scatter(x='percentUninsured', y='DeathsPerCapita', alpha=0.4)
data_mod.plot.scatter(x='percentBlack', y='DeathsPerCapita', alpha=0.4)
plt.show()

data.plot.scatter(x='lifeExpectancy', y='DeathsPerCapita')
data.plot.scatter(x='percentPhysicallyInactive', y='DeathsPerCapita')
plt.show()

data.plot.scatter(x='PopDensity', y='DeathsPerCapita')
data.plot.scatter(x='PopDensity', y='CasesPerCapita')
plt.show()

data.plot.scatter(x='CasesPerCapita', y='DeathsPerCapita', alpha=0.3)
plt.show()

########################################################################
# Classification modeling to predict whether county deaths/capita was  #
# higher than 0.002 with all variables.
########################################################################

#binning DeathsPerCapita to create target casslification varibale
data['DeathsPerCapita'].describe()

bin_edges = [-1,0.002,1]

data['Deaths_TopThird'] = pd.cut(data['DeathsPerCapita'],
                                    bins = bin_edges,
                                    right=True,
                                    ordered=False,
                                    labels=[0,1])

data['Deaths_TopThird'].value_counts()



#removing categorical and target variables and dropping rows with missing values
data_mod = data.drop(['FIPSCode', 'winner','area','TotalCases','TotalDeaths', 'DeathsPerCapita',
                    'stateAbbrev','countyName','DeathRate','CasesPerCapita','priceRecordedDate'], axis=1).dropna()
data_mod['waterViolationPresence'] = data_mod['waterViolationPresence'].astype('float64')               
data_mod.shape
data_mod.columns
data_mod.dtypes

#data_mod covers 317723378 people, or 96% of totalPopulation in data
#data_mod['totalPopulation'].sum()
#data['totalPopulation'].sum()

#creating trainign and vlaidation data
data_x = data_mod.drop(['Deaths_TopThird'], axis=1)
data_y = data_mod.loc[:,'Deaths_TopThird']
data_x.shape
data_y.shape

x_train, x_test, y_train, y_test = train_test_split(data_x,
                                                    data_y,
                                                    train_size=0.60)
x_train.shape
y_train.value_counts()
y_test.value_counts()

#Randomly oversampling positive class to balance trianing data
data_train_all = pd.concat([x_train, y_train], axis=1)
data_train_all.shape
data_train_all['Deaths_TopThird'].value_counts()

data_train_1 = data_train_all[data_train_all['Deaths_TopThird']==1]
data_train_1.shape

data_train_oversampled = data_train_all.append(data_train_1.sample(frac=0.7, replace=False))
data_train_oversampled.shape
data_train_oversampled['Deaths_TopThird'].value_counts()

x_train = data_train_oversampled.drop(['Deaths_TopThird'], axis=1)
x_train.shape
y_train = data_train_oversampled['Deaths_TopThird']
y_train.shape

y_train.value_counts()
y_test.value_counts()

#creating RF Classifier model, fitting it, and getting predictions
dt_all = RandomForestClassifier(n_estimators = 1000)

dt_all.fit(x_train, y_train)
dt_all_pred = dt_all.predict(x_test)

#prinint model assessment statistics
print(classification_report(y_test, dt_all_pred))
print(confusion_matrix(y_test, dt_all_pred))
metrics.roc_auc_score(y_test, dt_all_pred)

#Creating function to collect and aggregate split info from RF model
def getSplitInfo(dt):
    df = pd.DataFrame({'Variable' : data_x.columns.values})
    df['Node0_SplitCount'] = 0
    df['Node0_AvgThreshold'] = 0.000
    df['Node1_SplitCount'] = 0
    df['Node1_AvgThreshold'] = 0.000
    df['Node2_SplitCount'] = 0
    df['Node2_AvgThreshold'] = 0.000
    df['Node0_Running'] = 0.000
    df['Node1_Running'] = 0.000
    df['Node2_Running'] = 0.000
    df['Node0_SamplesLeft'] = 0
    df['Node1_SamplesLeft'] = 0
    df['Node2_SamplesLeft'] = 0
    df['Node0_SamplesRight'] = 0
    df['Node1_SamplesRight'] = 0
    df['Node2_SamplesRight'] = 0
    df['Node0_SamplesLeft_R'] = 0
    df['Node1_SamplesLeft_R'] = 0
    df['Node2_SamplesLeft_R'] = 0
    df['Node0_SamplesRight_R'] = 0
    df['Node1_SamplesRight_R'] = 0
    df['Node2_SamplesRight_R'] = 0

    for i in range(0,len(dt.estimators_)-1) :
        l = dt.estimators_[i].tree_.children_left[0]
        r = dt.estimators_[i].tree_.children_right[0]
        ll = dt.estimators_[i].tree_.children_left[l]
        lr = dt.estimators_[i].tree_.children_right[l]
        rl = dt.estimators_[i].tree_.children_left[r]
        rr = dt.estimators_[i].tree_.children_right[r]

        l_num = dt.estimators_[i].tree_.children_left[0]
        r_num = dt.estimators_[i].tree_.children_right[0]
        ll_num = dt.estimators_[i].tree_.children_left[l]
        lr_num = dt.estimators_[i].tree_.children_right[l]
        rl_num = dt.estimators_[i].tree_.children_left[r]
        rr_num = dt.estimators_[i].tree_.children_right[r]

        split_feature_0 = dt.estimators_[i].tree_.feature[0]
        split_feature_1 = dt.estimators_[i].tree_.feature[l]
        split_feature_2 = dt.estimators_[i].tree_.feature[r]

        if split_feature_0 >= 0 :
            df['Node0_SplitCount'][split_feature_0] += 1 
        if split_feature_1 >= 0 :
            df['Node1_SplitCount'][split_feature_1] += 1 
        if split_feature_2 >= 0 :
            df['Node2_SplitCount'][split_feature_2] += 1 

        if dt.estimators_[i].tree_.children_left[0] != dt.estimators_[i].tree_.children_right[0] :
            threshold_0 = dt.estimators_[i].tree_.threshold[0]
            df['Node0_Running'][split_feature_0] = df['Node0_Running'][split_feature_0] + threshold_0
            df['Node0_AvgThreshold'][split_feature_0] = df['Node0_Running'][split_feature_0] / df['Node0_SplitCount'][split_feature_0]
            df['Node0_SamplesLeft_R'][split_feature_0] = df['Node0_SamplesLeft_R'][split_feature_0] + dt.estimators_[i].tree_.n_node_samples[l]
            df['Node0_SamplesRight_R'][split_feature_0] = df['Node0_SamplesRight_R'][split_feature_0] + dt.estimators_[i].tree_.n_node_samples[r]
            df['Node0_SamplesLeft'][split_feature_0] = df['Node0_SamplesLeft_R'][split_feature_0] / df['Node0_SplitCount'][split_feature_0]
            df['Node0_SamplesRight'][split_feature_0] = df['Node0_SamplesRight_R'][split_feature_0] / df['Node0_SplitCount'][split_feature_0]

        if dt.estimators_[i].tree_.children_left[l] != dt.estimators_[i].tree_.children_right[l] :
            threshold_1 = dt.estimators_[i].tree_.threshold[l]
            df['Node1_Running'][split_feature_1] = df['Node1_Running'][split_feature_1] + threshold_1
            df['Node1_AvgThreshold'][split_feature_1] = df['Node1_Running'][split_feature_1] / df['Node1_SplitCount'][split_feature_1]
            df['Node1_SamplesLeft_R'][split_feature_1] = df['Node1_SamplesLeft_R'][split_feature_1] + dt.estimators_[i].tree_.n_node_samples[ll]
            df['Node1_SamplesRight_R'][split_feature_1] = df['Node1_SamplesRight_R'][split_feature_1] + dt.estimators_[i].tree_.n_node_samples[lr]
            df['Node1_SamplesLeft'][split_feature_1] = df['Node1_SamplesLeft_R'][split_feature_1] / df['Node1_SplitCount'][split_feature_1]
            df['Node1_SamplesRight'][split_feature_1] = df['Node1_SamplesRight_R'][split_feature_1] / df['Node1_SplitCount'][split_feature_1]

        if dt.estimators_[i].tree_.children_left[r] != dt.estimators_[i].tree_.children_right[r] :
            threshold_2 = dt.estimators_[i].tree_.threshold[r]
            df['Node2_Running'][split_feature_2] = df['Node2_Running'][split_feature_2] + threshold_2
            df['Node2_AvgThreshold'][split_feature_2] = df['Node2_Running'][split_feature_2] / df['Node2_SplitCount'][split_feature_2]
            df['Node2_SamplesLeft_R'][split_feature_2] = df['Node2_SamplesLeft_R'][split_feature_2] + dt.estimators_[i].tree_.n_node_samples[rl]
            df['Node2_SamplesRight_R'][split_feature_2] = df['Node2_SamplesRight_R'][split_feature_2] + dt.estimators_[i].tree_.n_node_samples[rr]
            df['Node2_SamplesLeft'][split_feature_2] = df['Node2_SamplesLeft_R'][split_feature_2] / df['Node2_SplitCount'][split_feature_2]
            df['Node2_SamplesRight'][split_feature_2] = df['Node2_SamplesRight_R'][split_feature_2] / df['Node2_SplitCount'][split_feature_2]

    df['PercentTopSplit'] = df['Node0_SplitCount'] / sum(df['Node0_SplitCount'])

    df['TimesInTop3'] = df['Node0_SplitCount'] + df['Node1_SplitCount'] + df['Node2_SplitCount'] 
    df['PercentinTop3'] = df['TimesInTop3'] / sum(df['Node0_SplitCount'])

    return(df)

#getting split info
vi_all = getSplitInfo(dt_all)

#Getting and printing variable importance from built-in method
importances = pd.DataFrame({'Variable' : data_x.columns.values,
                            'Importance' : dt_all.feature_importances_})

print(importances.sort_values('Importance', ascending=False))

#printing split info to console
print(vi_all[['Variable','TimesInTop3', 'PercentinTop3']].sort_values('PercentinTop3', ascending=False))

########################################################################
# Classification modeling to predict whether county deaths/capita was  #
# higher than 0.002 using msot important variabels from previous model #
# This is final Random Forest model                                    #     
########################################################################

#Selecting most important varibaels from previous model + total population temporarily
data_selected = data[['Deaths_TopThird','medianHousingPrice','proportionVotesWon_R','lifeExpectancy',
                    'percentPhysicallyInactive','percentChildrenInPoverty','percentFairPoorHealth',
                    'medianHouseholdIncome','percentNonHispanicWhite','percentUninsured',
                    'highSchoolGraduationRate', 'teenBirthRate','totalPopulation']]
data_selected.shape

#dropping rows with missing values. This model data will lose fewer rows than the model
#with all variabels included because there are fewers rows with NAs to be dropped
data_mod = data_selected.dropna()
data_mod.shape

#data_mod covers 323468481 people, or 98.5% of totalPopulation in data_all
data_mod['totalPopulation'].sum()
data['totalPopulation'].sum()

#Creating training and validation data
data_x = data_mod.drop(['Deaths_TopThird', 'totalPopulation'], axis=1)

data_y = data_mod.loc[:,'Deaths_TopThird']
data_x.shape
data_y.shape

x_train, x_test, y_train, y_test = train_test_split(data_x,
                                                    data_y,
                                                    train_size=0.60)
x_train.shape
y_train.value_counts()
y_test.value_counts()

#Randomly oversamplign positive class to balance training data only
data_train_all = pd.concat([x_train, y_train], axis=1)
data_train_all.shape
data_train_all['Deaths_TopThird'].value_counts()

data_train_1 = data_train_all[data_train_all['Deaths_TopThird']==1]
data_train_1.shape

data_train_oversampled = data_train_all.append(data_train_1.sample(frac=0.7, replace=False))
data_train_oversampled.shape
data_train_oversampled['Deaths_TopThird'].value_counts()

x_train = data_train_oversampled.drop(['Deaths_TopThird'], axis=1)
x_train.shape
y_train = data_train_oversampled['Deaths_TopThird']
y_train.shape

y_train.value_counts()
y_test.value_counts()

#Printing statistics for variables in x_train and x_test
for col in x_train.columns.values:
    print(col," - Mean:", np.mean(x_train[col]))
    print(col," - Median:", np.median(x_train[col]))
    print(col," - StDev:", np.std(x_train[col]))

for col in x_test.columns.values:
    print(col," - Mean:", np.mean(x_test[col]))
    print(col," - Median:", np.median(x_test[col]))
    print(col," - StDev:", np.std(x_test[col]))

#Creating model object. max_depth set to 8 because of improved accuracy compared to unlimited depth
#Exact number of 8 from trial and error
dt = RandomForestClassifier(n_estimators = 1000,
                            max_depth=8,
                            max_samples = 1800
                            )

#fitting model and producing predictions for validation data
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)

#printing model assessments
print(classification_report(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))
metrics.roc_auc_score(y_test, dt_pred)

#Aggregating top level splits and thresholds from Random Forest model
var_importances = getSplitInfo(dt)

#Getting and printing variable importance from built-in method
importances = pd.DataFrame({'Variable' : data_x.columns.values,
                            'Importance' : dt.feature_importances_})

print(importances.sort_values('Importance', ascending=False))

#printing split info to console
print(var_importances[['Variable','TimesInTop3', 'PercentinTop3']].sort_values('PercentinTop3', ascending=False))
print(var_importances[['Variable','Node0_SplitCount', 'Node0_AvgThreshold', 'PercentTopSplit']])
print(var_importances[['Variable','Node1_SplitCount', 'Node1_AvgThreshold']])
print(var_importances[['Variable','Node2_SplitCount', 'Node2_AvgThreshold']])

#writing split counts and thresholds to a csv
var_importances.to_csv(r'feature_importances.csv', sep=',')



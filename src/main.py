# %%
from script.formulation import *
from script.functions import *
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# %%
# Prepare for paths
filenames = get_selected_files()
ds_paths = dict(zip(filenames, get_paths(filenames, 'ds')))
df_paths = dict(zip(filenames, get_paths(filenames, 'df')))
log_paths = dict(zip(filenames, get_paths(filenames, 'log')))
test_samples = ['i160-314',
                'i160-245',
                'i160-313',
                'i160-242',
                'i160-241',
                'i160-244',
                'i160-343',
                'i160-344',
                'i160-341',
                'i160-345',
                'i160-342']

# %%
# Read dataframes
train_list = []
test_list = []
for file in filenames:
    tmp_df, runtime = dataframe_generate(ds_paths[file], log_paths[file])
    if file in test_samples:
        test_list.append(tmp_df)
    else:
        train_list.append(tmp_df)
df_train = pd.concat(train_list)
df_test = pd.concat(test_list)

# %%
# Prepare the train, test set for Evaluation 1
x_train, y_train = split_x_y(df_train)
x_test, y_test = split_x_y(df_test)

# %%
clfs = {
    "Support Vector Machine" : SVC(
        class_weight='balanced', probability=True, random_state=0),
    "Random Forest" : RandomForestClassifier(class_weight='balanced'),
    "Logistic Regression" : LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=0),
}

# %%
# for clf in clfs:
#     clfs[clf].fit(x_train, y_train)
# plt.hist(clfs["Support Vector Machine"].predict_proba(x_test)[:,1], bins=20)
# plt.hist(clfs["Random Forest"].predict_proba(x_test)[:,1], bins=20)
# plt.hist(clfs["Logistic Regression"].predict_proba(x_test)[:,1], bins=20)

# %%
# Adjust thresholds for LR classifier
print("Logistic Regression:")
print("Training...")
clfs['Logistic Regression'].fit(x_train, y_train)
print("Train Finished")
print("Feature Importance:")
print(x_train.std()*clfs['Logistic Regression'].coef_[0])
print("Adjust Thresholds:")
thresholds = np.arange(0,1,0.02)
for threshold in thresholds:
    y_pred_proba = clfs['Logistic Regression'].predict_proba(x_test)
    y_pred = (y_pred_proba[:,1] >= threshold).astype('int')
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    print("Threshold:",np.round(threshold,2), 
        "FN rate:", np.round(fn/(fn+tp), 2),
        "Pruning Rate:", np.round(100*(fn+tn)/len(y_pred),2), '%')

# %%
# Evaluation on each samples in test_list (Evaluation 2)
thresholds = np.arrange(0.05, 0.5, 0.05)
clf = clfs['Logistic Regression']
for filename in test_samples:
    ds_path = ds_paths[filename]
    log_path = log_paths[filename]
    for threshold in thresholds:
        solve(filename, clf, ds_path, log_path, threshold)

# %%




# from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_digits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder    ###For one hot encoding

#Importing Classifiers
from sklearn.linear_model import LogisticRegression    #All classifiers
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

############################################MNIST Data Base Code############################################
#Loading MNIST Data
mnist = load_digits()
#Splitting and Testing and Training Data
img_train, img_test, lbl_train, lbl_test=train_test_split(mnist.data, mnist.target, test_size=0.15, random_state=0, shuffle=True)


# Training Logistic Regression 
clf=LogisticRegression(solver = 'lbfgs')
clf.fit(img_train, lbl_train)
lbl_predictions_train=clf.predict(img_train)
lbl_predictions_test = clf.predict(img_test)

#Confusion Matrix for training data
cm_train = confusion_matrix(lbl_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(lbl_test, lbl_predictions_test)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]  #To nomalize the confusion matrix
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]

d_cm_test_lr=np.diagonal(cm_test)
d_cm_train_lr=np.diagonal(cm_train)


lrm_acc_score_test = accuracy_score(lbl_test, lbl_predictions_test)
lrm_acc_score_train = accuracy_score(lbl_train, lbl_predictions_train)

####Plotting Confusion Matrix for LR
f = plt.figure(figsize=(20,8))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(lrm_acc_score_train)
ax.set_title('MNIST Training Data Confusion Matrix Using LR', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0:2f}'.format(lrm_acc_score_test)
ax.set_title('MNIST Testing Data Confusion Matrix Using LR', size = 15);
plt.show()

# #Classification report for logistic regression
print('LR Metrics Class Wise')
report = classification_report(lbl_test, lbl_predictions_test)
print(report)


#print("Top 1 Accuracy Score of Test Data Logistic Regression= {0:.2f}".format( lrm_acc_score_test))
#print("Top 1 Accuracy Score of Training Data Logistic Regression= {0:.2f}".format( lrm_acc_score_train))


#Top 3 classes
prob_est_sample=clf.predict_proba(img_test)
topk_list=[]
for row in prob_est_sample:
    ids=[]
    p=0
    sort=sorted(row, reverse=True)
    while p < 3:     #Class identifying top 3 classes
        id=np.where(row==sort[p])
        ids.append(id[0][0])
        p=p+1
    topk_list.append(ids)
topk_class=np.asarray(topk_list)


#Accuracy score of top 3 classes
acc_score_k_test=0
for i in range(len(lbl_test)):
    if(lbl_test[i] in topk_class[i]):
        acc_score_k_test=acc_score_k_test+1
lrm_perc_accscorek_test=acc_score_k_test/len(lbl_test)
#print('Top 3 Accuracy score of Testing data Using Logistic Regression is {0:.2f}'.format( perc_acc_score_k_test))
lrm_acc=[lrm_acc_score_test, lrm_perc_accscorek_test ]



# Fitting and modelling SVM
clf = SVC(probability=True)
clf.fit(img_train, lbl_train)
 

clf.fit(img_train, lbl_train)
lbl_predictions_train=clf.predict(img_train)
lbl_predictions_test = clf.predict(img_test)

#Confusion Matrix for training data
cm_train = confusion_matrix(lbl_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(lbl_test, lbl_predictions_test)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]


d_cm_test_svm=np.diagonal(cm_test)
d_cm_train_svm=np.diagonal(cm_train)


svmm_acc_score_test = accuracy_score(lbl_test, lbl_predictions_test)
svmm_acc_score_train = accuracy_score(lbl_train, lbl_predictions_train)


###Plotting Confusion Matrix
f = plt.figure(figsize=(20,8))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(svmm_acc_score_train)
ax.set_title('MNIST Training Data Confusion Matrix Using SVM', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(svmm_acc_score_test)
ax.set_title('MNIST Testing Data Confusion Matrix Using SVM', size = 15);
plt.show()

# #Classification report using SVm
print('SVM Metrics Class Wise')
report = classification_report(lbl_test, lbl_predictions_test)
print(report)


#Top 3 classes
prob_est_sample=clf.predict_proba(img_test)
topk_list=[]
for row in prob_est_sample:
    ids=[]
    p=0
    sort=sorted(row, reverse=True)
    while p < 3:     #Class identifying top 3 classes
        id=np.where(row==sort[p])
        ids.append(id[0][0])
        p=p+1
    topk_list.append(ids)
topk_class=np.asarray(topk_list)


#Accuracy score of top 3 classes
acc_score_k_test=0
for i in range(len(lbl_test)):
    if(lbl_test[i] in topk_class[i]):
        acc_score_k_test=acc_score_k_test+1
svmm_perc_accscorek_test=acc_score_k_test/len(lbl_test)
#print('Top 3 Accuracy score of Testing data Using SVM is {0:.2f}'.format( perc_acc_score_k_test))
#print('Top 1 Accuracy Score of Test Data using SVM= {0:.2f}'.format(svmm_acc_score_test))
#print('Top 1 Accuracy Score of Training Data using SVM={0:.2f}'.format( svmm_acc_score_train))
svmm_acc=[svmm_acc_score_test, svmm_perc_accscorek_test ]


#### Plotting Top1 and Top3 Accuracy of SVM and LR

#f=plt.figure(figsize=(15,5))

f, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5) )
#f=plt.figure(figsize=(10,5))
#ax=f.add_subplot(1,2,1)
ax1.barh(['top1', 'top3'],lrm_acc )
ax1.set_ylabel('Category of Class ', fontsize=10)
ax1.set_xlabel('Accuracy', fontsize=10)
ax1.set_title('Top1 and Top3 Accuracy of Logistic Regression', size=15)
for i, v in enumerate(lrm_acc):
    ax1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')


#ax=f.add_subplot(1,2,2)
#f=plt.figure(figsize=(10,5))
ax2.barh(['top1', 'top3'],svmm_acc )
ax2.set_ylabel('Category of Accuracy', fontsize=10)
ax2.set_xlabel('Accuracy', fontsize=10)
#plt.xticks(indx, label, fontsize=5, rotation=30)
ax2.set_title('Top1 and Top3 Accuracy Support Vector Machine', size=15)

for i, v in enumerate(svmm_acc):
    ax2.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')


plt.show()



# Training KNN
clf = KNeighborsClassifier(n_neighbors = 1)
clf.fit(img_train, lbl_train)

clf.fit(img_train, lbl_train)
lbl_predictions_train=clf.predict(img_train)
lbl_predictions_test = clf.predict(img_test)

#Confusion Matrix for training data
cm_train = confusion_matrix(lbl_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(lbl_test, lbl_predictions_test)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]

d_cm_test_knn=np.diagonal(cm_test)
d_cm_train_knn=np.diagonal(cm_train)

knnm_acc_score_test = accuracy_score(lbl_test, lbl_predictions_test)
knnm_acc_score_train = accuracy_score(lbl_train, lbl_predictions_train)

####Plotting Confusion Matrix##########
f = plt.figure(figsize=(20,8))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(knnm_acc_score_train)
ax.set_title('MNIST Training Data Confusion Matrix Using KNN', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(knnm_acc_score_test)
ax.set_title('MNIST Testing Data Confusion Matrix Using KNN', size = 15);
plt.show()



# #Printing Classification report using KNN
print('KNN Metrics Class Wise')
report = classification_report(lbl_test, lbl_predictions_test)
print(report)

print('Top 1 Accuracy Score of Test Data using KNN= {0:.2f}'.format( knnm_acc_score_test))
print('Top 1 Accuracy Score of Training Data using KNN= {0:.2f}'.format( knnm_acc_score_train))


# Training Decision Tree
clf = DecisionTreeClassifier()
clf.fit(img_train, lbl_train)

lbl_predictions_train=clf.predict(img_train)
lbl_predictions_test = clf.predict(img_test)

#Confusion Matrix for training data
cm_train = confusion_matrix(lbl_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(lbl_test, lbl_predictions_test)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]

d_cm_test_dt=np.diagonal(cm_test)
d_cm_train_dt=np.diagonal(cm_train)

dtm_acc_score_test = accuracy_score(lbl_test, lbl_predictions_test)
dtm_acc_score_train = accuracy_score(lbl_train, lbl_predictions_train)


#Plotting Confusion matrix on HeatMap
f = plt.figure(figsize=(20,8))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(dtm_acc_score_train)
ax.set_title('MNIST Training Data Confusion Matrix Using DT', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(dtm_acc_score_test)
ax.set_title('MNIST Testing Data Confusion Matrix Using DT', size = 15);
plt.show()


# #Printing Classification report using Decision Tree
print('DT Metrics Class Wise')
report = classification_report(lbl_test, lbl_predictions_test)
print(report)


print('Top 1 Accuracy Score of Test Data using DT= {0:.2f}'.format( dtm_acc_score_test))
print('Top 1 Accuracy Score of Training Data using DT={0:2f}'.format( dtm_acc_score_train))


#Plotting Accuracy for all model 
classes=list(range(10))
f = plt.figure(figsize=(15,10))

ax=f.add_subplot(2,2,1)
# Plotting all model accuracy collectively
ax.scatter(classes, d_cm_test_lr, label='Testing Data')
ax.scatter(classes, d_cm_train_lr, label= 'Training Data', marker='*')
plt.xticks(np.arange(0, 10, 1))
ax.set_xlabel('Classes')
ax.set_ylabel('Accuracy')
ax.legend(loc=3)
ax.set_title('LR Accuracy Class Wise', size = 15);

ax=f.add_subplot(2,2,2)
ax.scatter(classes, d_cm_test_svm, label='Testing Data')
ax.scatter(classes, d_cm_train_svm, label= 'Training Data', marker='*')
plt.xticks(np.arange(0, 10, 1))
ax.set_xlabel('Classes')
ax.set_ylabel('Accuracy')
ax.legend(loc=3)
ax.set_title('SVM Accuracy Class Wise', size = 15);


ax=f.add_subplot(2,2,3)
ax.scatter(classes, d_cm_test_dt, label='Testing Data')
ax.scatter(classes, d_cm_train_dt, label= 'Training Data', marker='*')
plt.xticks(np.arange(0, 10, 1))
ax.set_xlabel('Classes')
ax.set_ylabel('Accuracy')
ax.legend(loc=3)
ax.set_title('DT Accuracy Class Wise', size = 15);

ax=f.add_subplot(2,2,4)
# sns.heatmap(cm_train, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax, vmin=0, vmax=1)
ax.scatter(classes, d_cm_test_knn, label='Testing Data', cmap='Colormap' )
ax.scatter(classes, d_cm_train_knn, label= 'Training Data', cmap='Colormap', marker='*' )
plt.xticks(np.arange(0, 10, 1))
ax.set_xlabel('Classes')
ax.set_ylabel('Accuracy')
ax.legend(loc=3)
ax.set_title('KNN Accuracy Class Wise', size = 15);

plt.show()




####################################################Credit Card Database Code########################################




############################Credit Card Database###################################
os.chdir(r"C:\Users\Chanda Grover\Assignment2\input")


#Data Loading
data = pd.read_csv('UCI_Credit_Card.csv')
df = data.copy()
target = 'default.payment.next.month'


# #Data PreProcessing
edu = np.unique(df['EDUCATION'])

df["EDUCATION"] = df["EDUCATION"].map({0: 4, 1:1, 2:2, 3:3, 4:4, 5: 5, 6: 5}) 
#1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown
edu = np.unique(df['EDUCATION'])
sex = np.unique(df['SEX'])
mrg = np.unique(df['MARRIAGE'])
df.MARRIAGE = df.MARRIAGE.map({0:3, 1:1, 2:2, 3:3})   #3 for others, 1 for married, 2 for single
mrg = np.unique(df['MARRIAGE'])


# encode generation labels using one-hot encoding scheme
gen_ohe = OneHotEncoder(handle_unknown='ignore')
gen_sex_arr = gen_ohe.fit_transform(df[['SEX']]).toarray()
gen_sex_labels = list(sex)
gen_sex = pd.DataFrame(gen_sex_arr, columns=['Gender1', 'Gender2'])


# encode generation labels using one-hot encoding scheme

gen_ohe = OneHotEncoder(handle_unknown='ignore')
gen_edu_arr = gen_ohe.fit_transform(df[['EDUCATION']]).toarray()
gen_edu_labels = list(edu)
gen_edu = pd.DataFrame(gen_edu_arr, columns=['EDU1', 'EDU2', 'EDU3', 'EDU4', 'EDU5'])


# encode generation labels using one-hot encoding scheme
gen_ohe = OneHotEncoder(handle_unknown='ignore')
gen_mrg_arr = gen_ohe.fit_transform(df[['MARRIAGE']]).toarray()
gen_mrg_labels = list(mrg)
gen_mrg = pd.DataFrame(gen_mrg_arr, columns=['MARRIAGE1', 'MARRIAGE2','MARRIAGE3'])


data1=pd.DataFrame(df, columns = ['LIMIT_BAL','AGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6','default.payment.next.month' ])
frame = [data1,gen_sex, gen_edu, gen_mrg]
ndf=pd.concat(frame, axis=1)
predictors = ndf.columns.drop(target)

#Data Training
X = np.asarray(ndf[predictors])
Y = np.asarray(ndf[target])
X=pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)   

#Modelling and Reporting
clf = LogisticRegression()
clf.fit(X_train, y_train)

lbl_predictions_train=clf.predict(X_train)
lbl_predictions_test = clf.predict(X_test)


#Confusion Matrix for training data
cm_train = confusion_matrix(y_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(y_test, lbl_predictions_test)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]


lrc_acc_score_test=clf.score(X_test,y_test)
lrc_acc_score_train=clf.score(X_train,y_train)

#Plotting Confusion Matrix
f = plt.figure(figsize=(15,5))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(lrc_acc_score_train)
ax.set_title('CC Training Data CM Using Logistic Regression', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(lrc_acc_score_test)
ax.set_title('CC Testing Data CM Using Logistic Regression', size = 15);
plt.show()

print('LR Metrics Class Wise')
report = classification_report(y_test, lbl_predictions_test)
print(report)


print("Logistic Regression Training Data accuracy {0:.2f}".format(lrc_acc_score_train))
print("Logistic Regression Testing Data accuracy {0:.2f}".format(lrc_acc_score_test))

y_score_test=clf.decision_function(X_test)
fpr_test, tpr_test, thresholds_test= roc_curve(y_test, y_score_test)

y_score_train=clf.decision_function(X_train)
fpr_train, tpr_train, thresholds_train= roc_curve(y_train, y_score_train)


#Plotting ROC Curve
plt.figure()
lw = 2
plt.plot(fpr_train, tpr_train, color='darkorange',
         lw=lw, label='train')   
plt.plot(fpr_test, tpr_test, color='navy', lw=lw, linestyle='--', label='test')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve LR')
plt.legend(loc="lower right")
plt.show()

####SVM######
#Modelling and Fitting
clf=LinearSVC()
clf.fit(X_train,y_train)

lbl_predictions_train=clf.predict(X_train)
lbl_predictions_test = clf.predict(X_test)


#Confusion Matrix for training data
cm_train = confusion_matrix(y_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(y_test, lbl_predictions_test)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]

svmc_acc_score_test=clf.score(X_test,y_test)
svmc_acc_score_train=clf.score(X_train,y_train)

#plotting Heatmap for SVM
f = plt.figure(figsize=(15,5))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(svmc_acc_score_train)
ax.set_title('CC Training Data Confusion Matrix Using SVM', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(svmc_acc_score_test)
ax.set_title('CC Testing Data Confusion Matrix Using SVM', size = 15);
plt.show()

print('SVM Metrics Class Wise')
report = classification_report(y_test, lbl_predictions_test)
print(report)


print("SVM Training Data accuracy using LinearSVC {0:.2f}".format(svmc_acc_score_train))
print("SVM Testing Data accuracy using LinearSVC {0:.2f}".format(svmc_acc_score_test))



y_score_test=clf.decision_function(X_test)
fpr_test, tpr_test, thresholds_test= roc_curve(y_test, y_score_test)

y_score_train=clf.decision_function(X_train)
fpr_train, tpr_train, thresholds_train= roc_curve(y_train, y_score_train)


#ROC Curve for SVM
plt.figure()
lw = 2
plt.plot(fpr_train, tpr_train, color='darkorange',
         lw=lw, label='train')    
plt.plot(fpr_test, tpr_test, color='navy', lw=lw, linestyle='--', label='test')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve LR')
plt.legend(loc="lower right")
plt.show()



#############DEcision Tree#######################

#Modelling Decision Tree
clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

lbl_predictions_train=clf.predict(X_train)
lbl_predictions_test = clf.predict(X_test)


#Confusion Matrix for training data
cm_train = confusion_matrix(y_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(y_test, lbl_predictions_test)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]


dtc_acc_score_test=clf.score(X_test,y_test)
dtc_acc_score_train=clf.score(X_train,y_train)

#Plotting CM on heatmap for DT
f = plt.figure(figsize=(15,5))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(dtc_acc_score_train)
ax.set_title('CC Training Data Confusion Matrix Using Decision Tree', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(dtc_acc_score_test)
ax.set_title('CC Testing Data Confusion Matrix Using Decision Tree', size = 15);
plt.show()
#Printing Report
print('DT Metrics Class Wise')
report = classification_report(y_test, lbl_predictions_test)
print(report)


print("Decision Tree Training Data accuracy {0:.2f}".format(dtc_acc_score_train))
print("Decision Tree Testing Data accuracy {0:.2f}".format(dtc_acc_score_test))




##KNN Classifier######
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train,y_train)


lbl_predictions_train=clf.predict(X_train)
lbl_predictions_test = clf.predict(X_test)


#Confusion Matrix for training data
cm_train = confusion_matrix(y_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(y_test, lbl_predictions_test)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]

knnc_acc_score_test=clf.score(X_test,y_test)
knnc_acc_score_train=clf.score(X_train,y_train)

#Plotting CM for KNN
f = plt.figure(figsize=(15,5))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(knnc_acc_score_train)
ax.set_title('CC Training Data Confusion Matrix Using KNN', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax, vmin=0, vmax=1)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(knnc_acc_score_test)
ax.set_title('CC Testing Data Confusion Matrix Using KNN', size = 15);
plt.show()

print('KNN Metrics Class Wise')
report = classification_report(y_test, lbl_predictions_test)
print(report)



# acc_score_test=clf.score(X_test,y_test)
# acc_score_train=clf.score(X_train,y_train)

print("KNN Training Data accuracy {0:.2f}".format(knnc_acc_score_train))
print("KNN Testing Data accuracy {0:.2f}".format(knnc_acc_score_test))


#Accuracy of MNIST and Credit Card Data
plt.figure(figsize= (8,8))
plt.title('Top 1 Acuracy for MNIST and Credit Card Database Model Wise', size=15)
classifiers=['SVM','LR', 'DT', 'KNN']
m_acc_score_test= [svmm_acc_score_test, lrm_acc_score_test, dtm_acc_score_test, knnm_acc_score_test]
c_acc_score_test= [svmc_acc_score_test, lrc_acc_score_test, dtc_acc_score_test, knnc_acc_score_test]
plt.scatter(classifiers, m_acc_score_test, marker= "P", label='MNIST')
plt.legend(loc=0)
plt.scatter(classifiers, c_acc_score_test, marker= "X", label='Credit Card')
plt.legend(loc=0)
plt.xlabel('Models', size=10)
plt.ylabel('Accuracy Score', size=10)
plt.show()
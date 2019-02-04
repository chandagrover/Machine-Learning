
# coding: utf-8

# In[198]:


# from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_digits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder    ###For one hot encoding


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

# os.chdir(r"C:\Users\Chanda Grover\Assignment2\input")
# mnist = fetch_mldata('MNIST original', data_home ='mnist-original')
############################################MNIST Data Base Code############################################
mnist = load_digits()


# print(mnist.data.shape)
# print(mnist.target.shape)

img_train, img_test, lbl_train, lbl_test=train_test_split(mnist.data, mnist.target, test_size=0.15, random_state=0, shuffle=True)

# print(img_train.shape)
# print(img_test.shape)
# print(lbl_train.shape)
# print(lbl_test.shape)


# Training Logistic Regression 
clf=LogisticRegression(solver = 'lbfgs', multi_class='multinomial')
clf.fit(img_train, lbl_train)
lbl_predictions_train=clf.predict(img_train)
lbl_predictions_test = clf.predict(img_test)

#Confusion Matrix for training data
cm_train = confusion_matrix(lbl_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(lbl_test, lbl_predictions_test)

acc_score_test = accuracy_score(lbl_test, lbl_predictions_test)
acc_score_train = accuracy_score(lbl_train, lbl_predictions_train)



####Plotting Confusion Matrix
plt.suptitle('Que 1a) MNIST DataBase Performance Metrics Using Logistic Regression', fontsize=15)
f = plt.figure(figsize=(20,8))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_train)
ax.set_title('Training Data Confusion Matrix Using LR', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_test)
ax.set_title('Testing Data Confusion Matrix Using LR', size = 15);
plt.show()

# #RClassification report using logistic regression
report = classification_report(lbl_test, lbl_predictions_test)
print(report)


print('Top 1 Accuracy Score of Test Data Logistic Regression=', acc_score_test)
print('Top 1 Accuracy Score of Training Data Logistic Regression=', acc_score_train)


#Top 3 classes
prob_est_sample=clf.predict_proba(img_test)
# print(prob_est)
topk_list=[]
for row in prob_est_sample:
    ids=[]
    p=0
#     print(row)
    sort=sorted(row, reverse=True)
#     print(sort)
    while p < 3:     #Class identifying top 3 classes
        id=np.where(row==sort[p])
        ids.append(id[0][0])
        p=p+1
    topk_list.append(ids)
# print(topk_list)
topk_class=np.asarray(topk_list)
# print(topk_class)
# print(topk_class.shape)

#Accuracy score of top 3 classes
acc_score_k_test=0
for i in range(len(lbl_test)):
#     print(lbl_test[i])
#     print(topk_class[i])
    if(lbl_test[i] in topk_class[i]):
        acc_score_k_test=acc_score_k_test+1
# print(acc_score_k_test)
perc_acc_score_k_test=acc_score_k_test/len(lbl_test)
print('Top 3 Accuracy score of Testing data Using Logistic Regression is', perc_acc_score_k_test)




# Training SVM
# clf = SVC()
clf=LinearSVC()
clf.fit(img_train, lbl_train)
 

clf.fit(img_train, lbl_train)
lbl_predictions_train=clf.predict(img_train)
lbl_predictions_test = clf.predict(img_test)

#Confusion Matrix for training data
cm_train = confusion_matrix(lbl_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(lbl_test, lbl_predictions_test)




###Plotting Confusion Matrix
f = plt.figure(figsize=(20,8))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_train)
ax.set_title('Training Data Confusion Matrix Using SVM', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_test)
ax.set_title('Testing Data Confusion Matrix Using SVM', size = 15);
plt.show()

# #RClassification report using SVm
report = classification_report(lbl_test, lbl_predictions_test)
print(report)


print('Top 1 Accuracy Score of Test Data using SVM=', acc_score_test)
print('Top 1 Accuracy Score of Training Data using SVM=', acc_score_train)



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


acc_score_test = accuracy_score(lbl_test, lbl_predictions_test)
acc_score_train = accuracy_score(lbl_train, lbl_predictions_train)

####Pltting Confusion Matrix##########
f = plt.figure(figsize=(20,8))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_train)
ax.set_title('Training Data Confusion Matrix Using KNN', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_test)
ax.set_title('Testing Data Confusion Matrix Using KNN', size = 15);
plt.show()



# #RClassification report using KNN
report = classification_report(lbl_test, lbl_predictions_test)
print(report)


print('Top 1 Accuracy Score of Test Data using KNN=', acc_score_test)
print('Top 1 Accuracy Score of Training Data using KNN=', acc_score_train)

#Top 3 classes
prob_est_sample=clf.predict_proba(img_test)
# print(prob_est)
topk_list=[]
for row in prob_est_sample:
    ids=[]
    p=0
#     print(row)
    sort=sorted(row, reverse=True)
#     print(sort)
    while p < 3:     #Class identifying top 3 classes
        id=np.where(row==sort[p])
        ids.append(id[0][0])
        p=p+1
    topk_list.append(ids)
# print(topk_list)
topk_class=np.asarray(topk_list)
# print(topk_class)
# print(topk_class.shape)

#Accuracy score of top 3 classes
acc_score_k_test=0
for i in range(len(lbl_test)):
#     print(lbl_test[i])
#     print(topk_class[i])
    if(lbl_test[i] in topk_class[i]):
        acc_score_k_test=acc_score_k_test+1
# print(acc_score_k_test)
perc_acc_score_k_test=acc_score_k_test/len(lbl_test)
print('Top 3 Accuracy score of Testing data Using KNN is', perc_acc_score_k_test)




# Training Decision Tree
clf = DecisionTreeClassifier()
clf.fit(img_train, lbl_train)

clf.fit(img_train, lbl_train)
lbl_predictions_train=clf.predict(img_train)
lbl_predictions_test = clf.predict(img_test)

#Confusion Matrix for training data
cm_train = confusion_matrix(lbl_train, lbl_predictions_train)
#Confusion Matrix for testing data
cm_test = confusion_matrix(lbl_test, lbl_predictions_test)

acc_score_test = accuracy_score(lbl_test, lbl_predictions_test)
acc_score_train = accuracy_score(lbl_train, lbl_predictions_train)

f = plt.figure(figsize=(20,8))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_train)
ax.set_title('Training Data Confusion Matrix Using DT', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_test)
ax.set_title('Testing Data Confusion Matrix Using DT', size = 15);
plt.show()


# #RClassification report using Decision Tree
report = classification_report(lbl_test, lbl_predictions_test)
print(report)


print('Top 1 Accuracy Score of Test Data using DT=', acc_score_test)
print('Top 1 Accuracy Score of Training Data using DT=', acc_score_train)

#Top 3 classes
prob_est_sample=clf.predict_proba(img_test)
# print(prob_est)
topk_list=[]
for row in prob_est_sample:
    ids=[]
    p=0
#     print(row)
    sort=sorted(row, reverse=True)
#     print(sort)
    while p < 3:     #Class identifying top 3 classes
        id=np.where(row==sort[p])
        ids.append(id[0][0])
        p=p+1
    topk_list.append(ids)
# print(topk_list)
topk_class=np.asarray(topk_list)
# print(topk_class)
# print(topk_class.shape)

#Accuracy score of top 3 classes
acc_score_k_test=0
for i in range(len(lbl_test)):
#     print(lbl_test[i])
#     print(topk_class[i])
    if(lbl_test[i] in topk_class[i]):
        acc_score_k_test=acc_score_k_test+1
# print(acc_score_k_test)
# perc_acc_score_k_test=acc_score_k_test/len(lbl_test)
print('Top 3 Accuracy score of Testing data Using Desision Tree is', perc_acc_score_k_test)







####################################################Credit Card Database Code########################################





############################Credit Card Database###################################
os.chdir(r"C:\Users\Chanda Grover\Assignment2\input")

#Data Loading
data = pd.read_csv('UCI_Credit_Card.csv')
df = data.copy()
target = 'default.payment.next.month'
# print(df.columns)
# df.head()

# #Data PreProcessing
edu = np.unique(df['EDUCATION'])
# print(edu)
df["EDUCATION"] = df["EDUCATION"].map({0: 4, 1:1, 2:2, 3:3, 4:4, 5: 5, 6: 5}) 
#1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown
edu = np.unique(df['EDUCATION'])
# print(edu)

sex = np.unique(df['SEX'])
# sex

mrg = np.unique(df['MARRIAGE'])
# print(mrg)
df.MARRIAGE = df.MARRIAGE.map({0:3, 1:1, 2:2, 3:3})   #3 for others, 1 for married, 2 for single
mrg = np.unique(df['MARRIAGE'])
# print(mrg)

# encode generation labels using one-hot encoding scheme
# print(df.head())
gen_ohe = OneHotEncoder()
gen_sex_arr = gen_ohe.fit_transform(df[['SEX']]).toarray()
gen_sex_labels = list(sex)
gen_sex = pd.DataFrame(gen_sex_arr, columns=['Gender1', 'Gender2'])
# print(gen_sex.head())

# encode generation labels using one-hot encoding scheme

gen_edu_arr = gen_ohe.fit_transform(df[['EDUCATION']]).toarray()
gen_edu_labels = list(edu)
gen_edu = pd.DataFrame(gen_edu_arr, columns=['EDU1', 'EDU2', 'EDU3', 'EDU4', 'EDU5'])
# print(gen_edu.head())

# encode generation labels using one-hot encoding scheme

gen_mrg_arr = gen_ohe.fit_transform(df[['MARRIAGE']]).toarray()
gen_mrg_labels = list(mrg)
gen_mrg = pd.DataFrame(gen_mrg_arr, columns=['MARRIAGE1', 'MARRIAGE2','MARRIAGE3'])

# print(gen_mrg.head())
# df.columns

data1=pd.DataFrame(df, columns = ['LIMIT_BAL','AGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6','default.payment.next.month' ])
# print(data1.head())
frame = [data1,gen_sex, gen_edu, gen_mrg]
ndf=pd.concat(frame, axis=1)
# print(result.head())

# predictors = df.columns.drop(['ID', target])
predictors = ndf.columns.drop(target)
# print(predictors.shape)
# print(result.shape)

#Data Training
X = np.asarray(ndf[predictors])
# print(X.shape)
Y = np.asarray(ndf[target])
# print(y.shape)
X=pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)
# X.head()
# data=X[:300,:].transpose()

#Modelling and Reporting
clf = LogisticRegression()
clf.fit(X_train, y_train)

lbl_predictions_train=clf.predict(X_train)
lbl_predictions_test = clf.predict(X_test)


# prec_score_test=precision_score(y_test, lbl_predictions_test, average='weighted')
# print("Logistic Regression Testing Data Precision",prec_score_test)


#Confusion Matrix for training data
cm_train = confusion_matrix(y_train, lbl_predictions_train, normalize=True)
#Confusion Matrix for testing data
cm_test = confusion_matrix(y_test, lbl_predictions_test, normalize=True)


acc_score_test=clf.score(X_test,y_test)
acc_score_train=clf.score(X_train,y_train)
#acc_perc_test=acc_score_test/acc_score_test+acc_score_train

f = plt.figure(figsize=(15,5))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_train)
ax.set_title('Testing Data Confusion Matrix Using Logistic Regression', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_test)
ax.set_title('Testing Data Confusion Matrix Using Logistic Regression', size = 15);
plt.show()

report = classification_report(y_test, lbl_predictions_test)
print(report)


print("Logistic Regression Training Data accuracy",acc_score_train)
print("Logistic Regression Testing Data accuracy",acc_score_test)

y_score_test=clf.decision_function(X_test)
# print(y_score_test.shape)
fpr, tpr, thresholds= roc_curve(y_test, y_score_test)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw)    #label='ROC curve (area = %0.2f)' % roc_auc[2]
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve LR')
plt.legend(loc="lower right")
plt.show()

# clf=svm.LinearSVC(C=1.0,random_state=0)
# clf.fit(X_train,y_train)
# accuracy=clf.score(X_test,y_test)
# print("SVM using Linear SVC accuracy",accuracy)


clf=LinearSVC(C=1.0,random_state=0)
clf.fit(X_train,y_train)

lbl_predictions_train=clf.predict(X_train)
lbl_predictions_test = clf.predict(X_test)


#Confusion Matrix for training data
cm_train = confusion_matrix(y_train, lbl_predictions_train, normalize=True)
#Confusion Matrix for testing data
cm_test = confusion_matrix(y_test, lbl_predictions_test, normalize=True)

acc_score_test=clf.score(X_test,y_test)
acc_score_train=clf.score(X_train,y_train)


f = plt.figure(figsize=(15,5))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_train)
ax.set_title('Training Data Confusion Matrix Using SVM', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_test)
ax.set_title('Testing Data Confusion Matrix Using SVM', size = 15);
plt.show()

report = classification_report(y_test, lbl_predictions_test)
print(report)


print("SVM Training Data accuracy using LinearSVC",acc_score_train)
print("SVM Testing Data accuracy using LinearSVC",acc_score_test)



y_score_test=clf.decision_function(X_test)
# print(y_score_test.shape)
fpr, tpr, thresholds= roc_curve(y_test, y_score_test)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw)    #label='ROC curve (area = %0.2f)' % roc_auc[2]
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve SVM')
plt.legend(loc="lower right")
plt.show()



#############DEcision Tree#######################

clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

lbl_predictions_train=clf.predict(X_train)
lbl_predictions_test = clf.predict(X_test)


#Confusion Matrix for training data
cm_train = confusion_matrix(y_train, lbl_predictions_train, normalize=True)
#Confusion Matrix for testing data
cm_test = confusion_matrix(y_test, lbl_predictions_test, normalize=True)


acc_score_test=clf.score(X_test,y_test)
acc_score_train=clf.score(X_train,y_train)

f = plt.figure(figsize=(15,5))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_train)
ax.set_title('Training Data Confusion Matrix Using Decision Tree', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_test)
ax.set_title('Testing Data Confusion Matrix Using Decision Tree', size = 15);
plt.show()

report = classification_report(y_test, lbl_predictions_test)
print(report)


print("Decision Tree Training Data accuracy",acc_score_train)
print("Decision Tree Testing Data accuracy",acc_score_test)



##KNN Classifier######
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train,y_train)


lbl_predictions_train=clf.predict(X_train)
lbl_predictions_test = clf.predict(X_test)


#Confusion Matrix for training data
cm_train = confusion_matrix(y_train, lbl_predictions_train, normalize=True)
#Confusion Matrix for testing data
cm_test = confusion_matrix(y_test, lbl_predictions_test, normalize=True)


f = plt.figure(figsize=(15,5))
ax=f.add_subplot(1,2,1)
sns.heatmap(cm_train, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r',ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_train)
ax.set_title('Training Data Confusion Matrix Using KNN', size = 15);

ax=f.add_subplot(1,2,2)
sns.heatmap(cm_test, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r', ax= ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
'Accuracy Score: {0}'.format(acc_score_test)
ax.set_title('Testing Data Confusion Matrix Using KNN', size = 15);
plt.show()


report = classification_report(y_test, lbl_predictions_test)
print(report)



acc_score_test=clf.score(X_test,y_test)
acc_score_train=clf.score(X_train,y_train)
print("KNN Training Data accuracy",acc_score_train)
print("KNN Testing Data accuracy",acc_score_test)



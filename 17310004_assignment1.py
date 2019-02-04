#######Thanks to Shubham, Rahul, Shiv, Priyanka for helping and clarifying my errors #######


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:01:18 2018

@author: Chanda
"""

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.gridspec as GridSpec
from sklearn import model_selection
from sklearn.cross_validation import KFold


boston = datasets.load_boston()
df = pd.DataFrame(boston.data)
y = boston.target
X=boston.data

train_error=[]
test_error=[]
R2_score=[]


train_size=[.5,.6,.7,.8,.9,.95,.99]
for x in train_size:
    boston_X_train, boston_X_test, boston_y_train, boston_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
    boston_reg = linear_model.LinearRegression(normalize=True)
    boston_reg.fit(boston_X_train,boston_y_train)
    boston_y_test_pred = boston_reg.predict(boston_X_test)
    boston_y_train_pred = boston_reg.predict(boston_X_train)
    train_error.append(mean_squared_error(boston_y_train, boston_y_train_pred))
    test_error.append(mean_squared_error(boston_y_test, boston_y_test_pred))
    R2_score.append(r2_score(boston_y_train, boston_y_train_pred))
plt.plot(train_size, train_error , color='blue',marker='.', label='Training Error')
plt.xlabel('Training Samples[0-1]')
plt.ylabel('Error')
plt.plot(train_size, test_error, color='red', marker='.' ,label='Test Error')
plt.title('Que 1a) Plot of training error and test error with given training samples on boston dataset', fontsize=15, color='k')
plt.legend()
plt.show()


plt.scatter(train_size, R2_score, color='black', label='R2 Score')
plt.title('Que 1b) Plot of R2 Score with training samples on boston dataset', fontsize=15, color='k')
plt.xlabel('Training Samples[0-1]')
plt.ylabel('R2-Score')
plt.legend()
plt.show()

the_grid=GridSpec.GridSpec(2,4)
#fig = plt.figure(1)
#gs1 = GridSpec.GridSpec(2,4)
#ax_list = [fig.add_subplot(ss) for ss in gs1]
#print(ax_list)
fig=plt.figure(figsize=(15,7))
alpha=[0,0.01,0.1,1]
i=0
for a in alpha:
    train_error=[]
    test_error=[]
    train_size=[.5,.6,.7,.8,.9,.95,.99]
    for x in train_size:
        boston_X_train, boston_X_test, boston_y_train, boston_y_test = model_selection.train_test_split(df, y, test_size=1-x)
#        print(type(boston_X_train))
#        print(type(boston_y_train))
#        print(boston_X_train, boston_y_train)
        boston_reg = linear_model.Ridge(alpha=a, normalize=True)
        boston_reg.fit(boston_X_train,boston_y_train)
        boston_y_test_pred = boston_reg.predict(boston_X_test)
        boston_y_train_pred = boston_reg.predict(boston_X_train)
        train_error.append(mean_squared_error(boston_y_train, boston_y_train_pred))
        test_error.append(mean_squared_error(boston_y_test, boston_y_test_pred))
    
    plt.subplot(the_grid[0,i])
    plt.ylim((0,150))
    plt.suptitle('Que 1c) Training Error, Test Error and R2 Score wrt Training Samples on boston dataset', fontsize=15)
    plt.plot(train_size, train_error , color='blue',marker='.', label='Training Error')
    plt.plot(train_size, test_error, color='red', marker='.',label='Test Error')
    plt.ylabel('Error')
    plt.xlabel('Training Data[0-1]')
    plt.title('Alpha=%1.2f' %a)
    plt.legend()    
    i=i+1

        
        
fig=plt.figure(figsize=(15,7))
alpha=[0,0.01,0.1,1]
i=0
for a in alpha:
    R2_score=[]
    train_size=[.5,.6,.7,.8,.9,.95,.99]
    for x in train_size:
        boston_X_train, boston_X_test, boston_y_train, boston_y_test = model_selection.train_test_split(df, y, test_size=1-x)
        boston_reg = linear_model.Ridge(alpha=a, normalize=True)
        boston_reg.fit(boston_X_train,boston_y_train)
        boston_y_test_pred = boston_reg.predict(boston_X_test)
        boston_y_train_pred = boston_reg.predict(boston_X_train)
        R2_score.append(r2_score(boston_y_train, boston_y_train_pred))
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    plt.scatter(train_size, R2_score, color='black', label='R2 Score')
    plt.xlabel('Training Data[0-1]')
    plt.ylabel('R2-Score')
    plt.title('Alpha=%1.2f' %a)

    plt.legend()
    i=i+1
#plt.legend()
               
        
#################################Question 2
fig=plt.figure(figsize=(15,7))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    error=[]
    valid_error=[]
    alpha=[0.0,0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5]        
    for a in alpha:
        boston_X_train, boston_X_test, boston_y_train, boston_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
#        print(boston_X_train.shape)
#        print(boston_y_train.shape)
        boston_reg = linear_model.Ridge(alpha=a, normalize=True)
#        cv = KFold(n_samples, n_folds=5, shuffle=False)
#        boston_reg=linear_model.RidgeCV(alpha, normalize=False, fit_intercept=True, cv=5, store_cv_values=True  )
        #With cv=5 and store_cv_values=True getting incompatible results, So using cross_validate function to cross_validate it.
        boston_reg.fit(boston_X_train,boston_y_train)
        boston_y_test_pred = boston_reg.predict(boston_X_test)
        boston_y_train_pred = boston_reg.predict(boston_X_train)
        train_error.append(mean_squared_error(boston_y_train, boston_y_train_pred))
        test_error.append(mean_squared_error(boston_y_test, boston_y_test_pred))
        kf=KFold(n=len(boston_X_train), n_folds=5, shuffle=True)
#        print (kf)
        mse=0
        for train_indices, test_indices in kf:
            boston_reg.fit(boston_X_train[train_indices], boston_y_train[train_indices])
            mse=mse+mean_squared_error(boston_y_train[test_indices], boston_reg.predict(boston_X_train[test_indices]))
#            valid_error.append(mse)
        valid_error.append(np.mean(mse)/5)
#        
        
#        cv_results=model_selection.cross_validate(boston_reg, boston_X_train, boston_y_train, cv=5)
#        error.append(cv_results['test_score'])
#        valid_error.append(np.mean(error))
#    print(valid_error)
    plt.subplot(the_grid[0,i])
    plt.ylim((0,150))
    plt.suptitle('Que 2) Training Error, Test Error, Validation Error and R2 Score wrt Alpha (Ridge Regression) on boston', fontsize=15)
    plt.plot(alpha, train_error , 'b',marker='.', label='Training Error')
    plt.plot(alpha, test_error, 'r', marker='.', label='Test Error')
    plt.plot(alpha, valid_error, 'r--' ,marker='.', linewidth=2, label='Validation Error')
    plt.xlabel('Alpha')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Training Data=%2.2f' %x)
    i=i+1


        
#print(boston_X_train)
        
fig=plt.figure(figsize=(15,7))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    R2_score=[]
    alpha=[0.0,0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5]    
    for a in alpha:
        boston_X_train, boston_X_test, boston_y_train, boston_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
        boston_reg = linear_model.Ridge(alpha=a, normalize=True)
        boston_reg.fit(boston_X_train,boston_y_train)
        boston_y_test_pred = boston_reg.predict(boston_X_test)
        boston_y_train_pred = boston_reg.predict(boston_X_train)
        R2_score.append(r2_score(boston_y_train, boston_y_train_pred))
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    
    plt.scatter(alpha, R2_score, color='black', label='R2 Score')
    plt.xlabel('Alpha')
    plt.ylabel('R2 Score')
    
    plt.title('Training Data=%2.2f' %x)
    plt.legend()
    i=i+1
#plt.legend()
               
        
########################Question 3
fig=plt.figure(figsize=(15,7))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    error=[]
    valid_error=[]
    alpha=[0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5]    
    for a in alpha:
        boston_X_train, boston_X_test, boston_y_train, boston_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
        boston_reg = linear_model.Lasso(alpha=a, normalize=True)
        boston_reg.fit(boston_X_train,boston_y_train)
        boston_y_test_pred = boston_reg.predict(boston_X_test)
        boston_y_train_pred = boston_reg.predict(boston_X_train)
        train_error.append(mean_squared_error(boston_y_train, boston_y_train_pred))
        test_error.append(mean_squared_error(boston_y_test, boston_y_test_pred))
        kf=KFold(n=len(boston_X_train), n_folds=5, shuffle=True)
#        print (kf)
        mse=0
        for train_indices, test_indices in kf:
            #print(boston_X_train.shape, boston_X_test.shape, boston_y_train.shape, boston_y_test.shape)
            #print(train_indices.shape,train_indices.shape)
            boston_reg.fit(boston_X_train[train_indices], boston_y_train[train_indices])
            #break
            mse=mse+mean_squared_error(boston_y_train[test_indices], boston_reg.predict(boston_X_train[test_indices]))
#        valid_error.append(mse)
        valid_error.append(np.mean(mse)/5)
        ###With Cross Validation
#        cv_results=model_selection.cross_validate(boston_reg, boston_X_train, boston_y_train, cv=5)
#        error.append(cv_results['test_score'])
#        valid_error.append(np.mean(error))
#        error.append(model_selection.cross_val_score(boston_reg, boston_X_train, boston_y_train, cv=5))    
#        valid_error.append(1-np.mean(error))
    
    plt.subplot(the_grid[0,i])
    plt.ylim((0,150))
    plt.suptitle(' Que 3) Training Error, Test Error, Validation Error and R2 Score wrt Alpha(LAsso Regression) on boston', fontsize=15)
    plt.plot(alpha, train_error , color='blue',marker='.', label='Training Error')
    plt.plot(alpha, test_error, color='red', marker='.',label='Test Error')
    plt.plot(alpha, valid_error, 'r--', marker='.', label='Validation Error')
    plt.xlabel('Alphas')
    plt.ylabel('Error')
    plt.title('Training=%2.2f' %x)
    plt.legend()    
    i=i+1

        
        
fig=plt.figure(figsize=(15,7))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    R2_score=[]
    alpha=[0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5]    
    for a in alpha:
        boston_X_train, boston_X_test, boston_y_train, boston_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
        boston_reg = linear_model.Lasso(alpha=a, normalize=True)
        boston_reg.fit(boston_X_train,boston_y_train)
        boston_y_test_pred = boston_reg.predict(boston_X_test)
        boston_y_train_pred = boston_reg.predict(boston_X_train)
        R2_score.append(r2_score(boston_y_train, boston_y_train_pred))
    
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    plt.scatter(alpha, R2_score, color='black', label='R2 Score')
    plt.xlabel('Alphas')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.title('Training=%2.2f' %x)
    i=i+1
    
    
    
    
#############################Question 4######################################

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:44:22 2018

@author: Chanda
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
import matplotlib.gridspec as GridSpec
from sklearn import model_selection
from sklearn.cross_validation import KFold


boston = datasets.load_boston()
df = pd.DataFrame(boston.data)
y = boston.target
X=boston.data
#print(X.shape[0])

the_grid=GridSpec.GridSpec(2,4)

#def column(matrix, i):
#    return [row[i] for row in matrix]


fig=plt.figure(figsize=(15,10))
alpha=[0,0.01,0.1,1]
i=0
for a in alpha:
    train_error=[]
    test_error=[]
    R2_score=[]
    train_size=[.5,.6,.7,.8,.9,.95,.99]
    for x in train_size:
        boston_X_train, boston_X_test, boston_y_train, boston_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
        
#        boston_X_train=preprocessing.normalize(boston_X_train_WN)
#        boston_X_test=preprocessing.normalize(boston_X_test_WN)
        X_min_train=np.min(boston_X_train, axis=0)
        X_max_train=np.max(boston_X_train, axis=0)
        boston_X_train=np.abs(boston_X_train-X_min_train)/(X_max_train - X_min_train)
        
        
        X_min_test=np.min(boston_X_test, axis=0)
        X_max_test=np.max(boston_X_test, axis=0)
        boston_X_test=np.abs(boston_X_test - X_min_test)/(X_max_test-X_min_test)

#        boston_y_train=preprocessing.normalize(boston_y_train)
        
       
#        y_train_min=np.min(boston_y_train)  
#        y_train_max=np.max(boston_y_train)
##        boston_y_train=np.abs(boston_y_train-y_train_min)/(y_train_max-y_train_min)
#        
#        y_test_min=np.min(boston_y_test)   
#        y_test_max=np.max(boston_y_test)
##        boston_y_test=np.abs(boston_y_test-y_test_min)/(y_test_max-y_test_min)
        
        
        X_T=np.transpose(boston_X_train)
        temp=np.matmul(X_T,boston_X_train)
        temp=temp+a*np.eye(temp.shape[0], dtype=int)
        a1=np.linalg.inv(temp)
        b=np.matmul(X_T, boston_y_train)
        w=np.matmul(a1,b)
        boston_y_train_pred=np.matmul(boston_X_train,w)
        boston_y_test_pred=np.matmul(boston_X_test,w)
        
        y_train_mean=np.mean(boston_y_train)   #boston_y_train_pred
        
#        y_train_pred_min=np.min(boston_y_train_pred)   #boston_y_train_pred
#        y_train_pred_max=np.max(boston_y_train_pred)
#        boston_y_train_pred=np.abs(boston_y_train_pred*(y_train_pred_max-y_train_pred_min))+(y_train_pred_min)
#        
#        y_test_pred_min=np.mean(boston_y_test_pred)    # boston_y_test_pred
#        y_test_pred_max=np.max(boston_y_test_pred)       
#        boston_y_test_pred=np.abs(boston_y_test_pred*(y_test_pred_max-y_test_pred_min))+(y_test_pred_min)

##        print(boston_y_train_pred)
#        print(boston_y_test_pred)
       
#        boston_y_train_pred=preprocessing.normalize(boston_y_train_pred_WN)
#        boston_y_test_pred=preprocessing.normalize(boston_y_test_pred_WN)
#        
        tot_var=np.sum((boston_y_train - y_train_mean)**2)/len(boston_y_train)
        error=np.sum((boston_y_train - boston_y_train_pred)**2)/len(boston_y_train)
        train_error.append(error)
        test_error.append(np.sum((boston_y_test - boston_y_test_pred)**2)/len(boston_y_test))
        R2_score.append(1-(error/tot_var))
#    print(train_error)
#    print(test_error)
#    print(R2_score)
    plt.subplot(the_grid[0,i])
    plt.ylim((0,150))
    plt.suptitle('Que 4a) Training Error, Test Error and R2 Score wrt Training Samples on boston dataset Without inbuilt function with normaliztion', fontsize=15)
    plt.plot(train_size, train_error , color='blue',marker='.', label='Training Error')
    plt.plot(train_size, test_error, color='red', marker='.',label='Test Error')
    plt.ylabel('Error')
    plt.xlabel('Training Data[0-1]')
    plt.title('Alpha=%1.2f' %a)
    plt.legend()    
   
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    plt.scatter(train_size, R2_score, color='black', label='R2 Score')
    plt.xlabel('Training Data[0-1]')
    plt.ylabel('R2-Score')
    plt.title('Alpha=%1.2f' %a)

    plt.legend()
    i=i+1
    
    
    
    
    ######################Ridge Regression with cross validation##############################
the_grid=GridSpec.GridSpec(2,4)

fig=plt.figure(figsize=(15,10))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    error=[]
    valid_error=[]
    R2_score=[]
    alpha=[0.0,0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5] 
    for a in alpha:
        boston_X_train, boston_X_test, boston_y_train, boston_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
#        X_mean_train=np.mean(boston_X_train, axis=0)
##        print(X_mean_train)
#        X_std_train=np.std(boston_X_train, axis=0)
##        print(X_std_train)
#        boston_X_train=(boston_X_train-X_mean_train)/X_std_train
#        
#        X_mean_test=np.mean(boston_X_test, axis=0)
#        X_std_test=np.std(boston_X_test, axis=0)
#        boston_X_test=(boston_X_test - X_mean_test)/X_std_test
#
        X_min_train=np.min(boston_X_train, axis=0)
        X_max_train=np.max(boston_X_train, axis=0)
        boston_X_train=np.abs(boston_X_train-X_min_train)/(X_max_train - X_min_train)
        
        
        X_min_test=np.min(boston_X_test, axis=0)
        X_max_test=np.max(boston_X_test, axis=0)
        boston_X_test=np.abs(boston_X_test - X_min_test)/(X_max_test-X_min_test)

        
        
        X_T=np.transpose(boston_X_train)
        temp=np.matmul(X_T,boston_X_train)
        temp=temp+a*np.eye(temp.shape[0], dtype=int)
        a1=np.linalg.inv(temp)
        b=np.matmul(X_T, boston_y_train)
        w=np.matmul(a1,b)
        boston_y_train_pred=np.matmul(boston_X_train,w)
        boston_y_test_pred=np.matmul(boston_X_test,w)
        y_train_mean=np.mean(boston_y_train)
        
        
#        y_tain_std=np.std(boston_y_train_pred)
#        y_test_mean=np.mean(boston_y_test_pred)    # boston_y_test_pred
#        y_test_std=np.std(boston_y_test_pred)
#        
#        boston_y_train_pred=(boston_y_train_pred*y_tain_std)+y_train_mean
#        boston_y_test_pred=(boston_y_test_pred*y_test_std)+y_test_mean
#      
        
        tot_var=np.sum((boston_y_train - y_train_mean)**2)/len(boston_y_train)
        error=np.sum((boston_y_train - boston_y_train_pred)**2)/len(boston_y_train)
        train_error.append(error)
        test_error.append(np.sum((boston_y_test - boston_y_test_pred)**2)/len(boston_y_test))
        R2_score.append(1-(error/tot_var))
        kf=KFold(n=len(boston_X_train), n_folds=5, shuffle=True)
        mse=0
        for train_indices, test_indices in kf:
            X_T=np.transpose(boston_X_train[train_indices])
            temp=np.matmul(X_T, boston_X_train[train_indices])
            temp=temp+a*np.eye(temp.shape[0], dtype=int)
            a=np.linalg.inv(temp)
            b=np.matmul(X_T, boston_y_train[train_indices] )
            w=np.matmul(a,b)
            X_train_predict=np.matmul(boston_X_train[test_indices], w)
            mse=mse+ np.sum((boston_y_train[test_indices] - X_train_predict)**2)/len(X_train_predict)
        valid_error.append(np.mean(mse)/5)
    plt.subplot(the_grid[0,i])
    plt.ylim((0,150))
    plt.suptitle('Que 4) Training Error, Test Error, validation Error and R2 Score wrt Alpha Without inbuilt function with normaliztion', fontsize=15)
    plt.plot(alpha, train_error , color='blue',marker='.', label='Training Error')
    plt.plot(alpha, test_error, color='red', marker='.',label='Test Error')
    plt.plot(alpha, valid_error, 'r--' ,marker='.', linewidth=2, label='Validation Error')
    
    plt.ylabel('Error')
    plt.xlabel('Alpha')
    plt.title('Training Data=%1.2f' %x)
    plt.legend()    
   
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    plt.scatter(alpha, R2_score, color='black', label='R2 Score')
    plt.xlabel('Alpha')
    plt.ylabel('R2-Score')
    plt.title('Training Data=%1.2f' %x)

    plt.legend()
    i=i+1
        
    

#####################################################################################################diabetes##########################################################################################################################################################################################################################
##############Que 5.1) ###########################
fig=plt.figure(figsize=(10,6))
diabetes= datasets.load_diabetes()
df = pd.DataFrame(diabetes.data)
y = diabetes.target
X=diabetes.data

train_error=[]
test_error=[]
R2_score=[]


train_size=[.5,.6,.7,.8,.9,.95,.99]
for x in train_size:
    diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
    diabetes_reg = linear_model.LinearRegression(normalize=True)
    diabetes_reg.fit(diabetes_X_train,diabetes_y_train)
    diabetes_y_test_pred = diabetes_reg.predict(diabetes_X_test)
    diabetes_y_train_pred = diabetes_reg.predict(diabetes_X_train)
    train_error.append(mean_squared_error(diabetes_y_train, diabetes_y_train_pred))
    test_error.append(mean_squared_error(diabetes_y_test, diabetes_y_test_pred))
    R2_score.append(r2_score(diabetes_y_train, diabetes_y_train_pred))
plt.plot(train_size, train_error , color='blue',marker='.', label='Training Error')
plt.xlabel('Training Samples[0-1]')
plt.ylabel('Error')
plt.plot(train_size, test_error, color='red', marker='.' ,label='Test Error')
plt.title('Que 5.1a) Plot of training error and test error with given training samples on diabetes dataset', fontsize=15, color='k')
plt.legend()
plt.show()


plt.scatter(train_size, R2_score, color='black', label='R2 Score')
plt.title('Que 5. 1b)Plot of R2 Score with training samples on diabetes dataset', fontsize=15, color='k')
plt.xlabel('Training Samples[0-1]')
plt.ylabel('R2-Score')
plt.legend()
plt.show()

the_grid=GridSpec.GridSpec(2,4)
#fig = plt.figure(1)
#gs1 = GridSpec.GridSpec(2,4)
#ax_list = [fig.add_subplot(ss) for ss in gs1]
#print(ax_list)
fig=plt.figure(figsize=(15,7))
alpha=[0,0.01,0.1,1]
i=0
for a in alpha:
    train_error=[]
    test_error=[]
    train_size=[.5,.6,.7,.8,.9,.95,.99]
    for x in train_size:
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = model_selection.train_test_split(df, y, test_size=1-x)
#        print(type(diabetes_X_train))
#        print(type(diabetes_y_train))
#        print(diabetes_X_train, diabetes_y_train)
        diabetes_reg = linear_model.Ridge(alpha=a, normalize=True)
        diabetes_reg.fit(diabetes_X_train,diabetes_y_train)
        diabetes_y_test_pred = diabetes_reg.predict(diabetes_X_test)
        diabetes_y_train_pred = diabetes_reg.predict(diabetes_X_train)
        train_error.append(mean_squared_error(diabetes_y_train, diabetes_y_train_pred))
        test_error.append(mean_squared_error(diabetes_y_test, diabetes_y_test_pred))
    
    plt.subplot(the_grid[0,i])
    plt.ylim((0,8000))
    plt.suptitle('Que 5.1cTraining Error, Test Error and R2 Score wrt Training Samples on diabetesdataset', fontsize=15)
    plt.plot(train_size, train_error , color='blue',marker='.', label='Training Error')
    plt.plot(train_size, test_error, color='red', marker='.',label='Test Error')
    plt.ylabel('Error')
    plt.xlabel('Training Data[0-1]')
    plt.title('Alpha=%1.2f' %a)
    plt.legend()    
    i=i+1

        
        
fig=plt.figure(figsize=(15,7))
alpha=[0,0.01,0.1,1]
i=0
for a in alpha:
    R2_score=[]
    train_size=[.5,.6,.7,.8,.9,.95,.99]
    for x in train_size:
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = model_selection.train_test_split(df, y, test_size=1-x)
        diabetes_reg = linear_model.Ridge(alpha=a, normalize=True)
        diabetes_reg.fit(diabetes_X_train,diabetes_y_train)
        diabetes_y_test_pred = diabetes_reg.predict(diabetes_X_test)
        diabetes_y_train_pred = diabetes_reg.predict(diabetes_X_train)
        R2_score.append(r2_score(diabetes_y_train, diabetes_y_train_pred))
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    plt.scatter(train_size, R2_score, color='black', label='R2 Score')
    plt.xlabel('Training Data[0-1]')
    plt.ylabel('R2-Score')
    plt.title('Alpha=%1.2f' %a)

    plt.legend()
    i=i+1
#plt.legend()
               
        
#################################Question 5.2) #################3
fig=plt.figure(figsize=(15,7))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    error=[]
    valid_error=[]
    alpha=[0.0,0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5]        
    for a in alpha:
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
#        print(diabetes_X_train.shape)
#        print(diabetes_y_train.shape)
        diabetes_reg = linear_model.Ridge(alpha=a, normalize=True)
#        cv = KFold(n_samples, n_folds=5, shuffle=False)
#        diabetes_reg=linear_model.RidgeCV(alpha, normalize=False, fit_intercept=True, cv=5, store_cv_values=True  )
        #With cv=5 and store_cv_values=True getting incompatible results, So using cross_validate function to cross_validate it.
        diabetes_reg.fit(diabetes_X_train,diabetes_y_train)
        diabetes_y_test_pred = diabetes_reg.predict(diabetes_X_test)
        diabetes_y_train_pred = diabetes_reg.predict(diabetes_X_train)
        train_error.append(mean_squared_error(diabetes_y_train, diabetes_y_train_pred))
        test_error.append(mean_squared_error(diabetes_y_test, diabetes_y_test_pred))
        kf=KFold(n=len(diabetes_X_train), n_folds=5, shuffle=True)
#        print (kf)
        mse=0
        for train_indices, test_indices in kf:
            diabetes_reg.fit(diabetes_X_train[train_indices], diabetes_y_train[train_indices])
            mse=mse+mean_squared_error(diabetes_y_train[test_indices], diabetes_reg.predict(diabetes_X_train[test_indices]))
#            valid_error.append(mse)
        valid_error.append(np.mean(mse)/5)
#        
        
#        cv_results=model_selection.cross_validate(diabetes_reg, diabetes_X_train, diabetes_y_train, cv=5)
#        error.append(cv_results['test_score'])
#        valid_error.append(np.mean(error))
#    print(valid_error)
    plt.subplot(the_grid[0,i])
    plt.ylim((0,8000))
    plt.suptitle('Que 5.2Training Error, Test Error, Validation Error and R2 Score wrt Alpha (Ridge Regression) on diabetes', fontsize=15)
    plt.plot(alpha, train_error , 'b',marker='.', label='Training Error')
    plt.plot(alpha, test_error, 'r', marker='.', label='Test Error')
    plt.plot(alpha, valid_error, 'r--' ,marker='.', linewidth=2, label='Validation Error')
    plt.xlabel('Alpha')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Training Data=%2.2f' %x)
    i=i+1


        
#print(diabetes_X_train)
        
fig=plt.figure(figsize=(15,7))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    R2_score=[]
    alpha=[0.0,0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5]    
    for a in alpha:
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
        diabetes_reg = linear_model.Ridge(alpha=a, normalize=True)
        diabetes_reg.fit(diabetes_X_train,diabetes_y_train)
        diabetes_y_test_pred = diabetes_reg.predict(diabetes_X_test)
        diabetes_y_train_pred = diabetes_reg.predict(diabetes_X_train)
        R2_score.append(r2_score(diabetes_y_train, diabetes_y_train_pred))
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    
    plt.scatter(alpha, R2_score, color='black', label='R2 Score')
    plt.xlabel('Alpha')
    plt.ylabel('R2 Score')
    
    plt.title('Training Data=%2.2f' %x)
    plt.legend()
    i=i+1
#plt.legend()
               
        
########################Question 5.3)#######################
fig=plt.figure(figsize=(15,7))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    error=[]
    valid_error=[]
    alpha=[0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5]    
    for a in alpha:
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
        diabetes_reg = linear_model.Lasso(alpha=a, normalize=True)
        diabetes_reg.fit(diabetes_X_train,diabetes_y_train)
        diabetes_y_test_pred = diabetes_reg.predict(diabetes_X_test)
        diabetes_y_train_pred = diabetes_reg.predict(diabetes_X_train)
        train_error.append(mean_squared_error(diabetes_y_train, diabetes_y_train_pred))
        test_error.append(mean_squared_error(diabetes_y_test, diabetes_y_test_pred))
        kf=KFold(n=len(diabetes_X_train), n_folds=5, shuffle=True)
#        print (kf)
        mse=0
        for train_indices, test_indices in kf:
            #print(diabetes_X_train.shape, diabetes_X_test.shape, diabetes_y_train.shape, diabetes_y_test.shape)
            #print(train_indices.shape,train_indices.shape)
            diabetes_reg.fit(diabetes_X_train[train_indices], diabetes_y_train[train_indices])
            #break
            mse=mse+mean_squared_error(diabetes_y_train[test_indices], diabetes_reg.predict(diabetes_X_train[test_indices]))
#        valid_error.append(mse)
        valid_error.append(np.mean(mse)/5)
        ###With Cross Validation
#        cv_results=model_selection.cross_validate(diabetes_reg, diabetes_X_train, diabetes_y_train, cv=5)
#        error.append(cv_results['test_score'])
#        valid_error.append(np.mean(error))
#        error.append(model_selection.cross_val_score(diabetes_reg, diabetes_X_train, diabetes_y_train, cv=5))    
#        valid_error.append(1-np.mean(error))
    
    plt.subplot(the_grid[0,i])
    plt.ylim((0,8000))
    plt.suptitle('Que 5.3) Training Error, Test Error, Validation Error and R2 Score wrt Alpha(LAsso Regression) on diabetes', fontsize=15)
    plt.plot(alpha, train_error , color='blue',marker='.', label='Training Error')
    plt.plot(alpha, test_error, color='red', marker='.',label='Test Error')
    plt.plot(alpha, valid_error, 'r--', marker='.', label='Validation Error')
    plt.xlabel('Alphas')
    plt.ylabel('Error')
    plt.title('Training=%2.2f' %x)
    plt.legend()    
    i=i+1

        
        
fig=plt.figure(figsize=(15,7))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    R2_score=[]
    alpha=[0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5]    
    for a in alpha:
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
        diabetes_reg = linear_model.Lasso(alpha=a, normalize=True)
        diabetes_reg.fit(diabetes_X_train,diabetes_y_train)
        diabetes_y_test_pred = diabetes_reg.predict(diabetes_X_test)
        diabetes_y_train_pred = diabetes_reg.predict(diabetes_X_train)
        R2_score.append(r2_score(diabetes_y_train, diabetes_y_train_pred))
    
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    plt.scatter(alpha, R2_score, color='black', label='R2 Score')
    plt.xlabel('Alphas')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.title('Training=%2.2f' %x)
    i=i+1

#############################Question 4######################################
fig=plt.figure(figsize=(15,10))
alpha=[0,0.01,0.1,1]
i=0
for a in alpha:
    train_error=[]
    test_error=[]
    R2_score=[]
    train_size=[.5,.6,.7,.8,.9,.95,.99]
    for x in train_size:
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
        
#        diabetes_X_train=preprocessing.normalize(diabetes_X_train_WN)
#        diabetes_X_test=preprocessing.normalize(diabetes_X_test_WN)
        X_min_train=np.min(diabetes_X_train, axis=0)
        X_max_train=np.max(diabetes_X_train, axis=0)
        diabetes_X_train=np.abs(diabetes_X_train-X_min_train)/(X_max_train - X_min_train)
        
        
        X_min_test=np.min(diabetes_X_test, axis=0)
        X_max_test=np.max(diabetes_X_test, axis=0)
        diabetes_X_test=np.abs(diabetes_X_test - X_min_test)/(X_max_test-X_min_test)

#        diabetes_y_train=preprocessing.normalize(diabetes_y_train)
        
       
#        y_train_min=np.min(diabetes_y_train)  
#        y_train_max=np.max(diabetes_y_train)
##        diabetes_y_train=np.abs(diabetes_y_train-y_train_min)/(y_train_max-y_train_min)
#        
#        y_test_min=np.min(diabetes_y_test)   
#        y_test_max=np.max(diabetes_y_test)
##        diabetes_y_test=np.abs(diabetes_y_test-y_test_min)/(y_test_max-y_test_min)
        
        
        X_T=np.transpose(diabetes_X_train)
        temp=np.matmul(X_T,diabetes_X_train)
        temp=temp+a*np.eye(temp.shape[0], dtype=int)
        a1=np.linalg.inv(temp)
        b=np.matmul(X_T, diabetes_y_train)
        w=np.matmul(a1,b)
        diabetes_y_train_pred=np.matmul(diabetes_X_train,w)
        diabetes_y_test_pred=np.matmul(diabetes_X_test,w)
        
        y_train_mean=np.mean(diabetes_y_train)   #diabetes_y_train_pred
        
#        y_train_pred_min=np.min(diabetes_y_train_pred)   #diabetes_y_train_pred
#        y_train_pred_max=np.max(diabetes_y_train_pred)
#        diabetes_y_train_pred=np.abs(diabetes_y_train_pred*(y_train_pred_max-y_train_pred_min))+(y_train_pred_min)
#        
#        y_test_pred_min=np.mean(diabetes_y_test_pred)    # diabetes_y_test_pred
#        y_test_pred_max=np.max(diabetes_y_test_pred)       
#        diabetes_y_test_pred=np.abs(diabetes_y_test_pred*(y_test_pred_max-y_test_pred_min))+(y_test_pred_min)

##        print(diabetes_y_train_pred)
#        print(diabetes_y_test_pred)
       
#        diabetes_y_train_pred=preprocessing.normalize(diabetes_y_train_pred_WN)
#        diabetes_y_test_pred=preprocessing.normalize(diabetes_y_test_pred_WN)
#        
        tot_var=np.sum((diabetes_y_train - y_train_mean)**2)/len(diabetes_y_train)
        error=np.sum((diabetes_y_train - diabetes_y_train_pred)**2)/len(diabetes_y_train)
        train_error.append(error)
        test_error.append(np.sum((diabetes_y_test - diabetes_y_test_pred)**2)/len(diabetes_y_test))
        R2_score.append(1-(error/tot_var))
#    print(train_error)
#    print(test_error)
#    print(R2_score)
    plt.subplot(the_grid[0,i])
    plt.ylim((0,8000))
    plt.suptitle('Que 4a) Training Error, Test Error and R2 Score wrt Training Samples on diabetes dataset Without inbuilt function with normaliztion', fontsize=15)
    plt.plot(train_size, train_error , color='blue',marker='.', label='Training Error')
    plt.plot(train_size, test_error, color='red', marker='.',label='Test Error')
    plt.ylabel('Error')
    plt.xlabel('Training Data[0-1]')
    plt.title('Alpha=%1.2f' %a)
    plt.legend()    
   
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    plt.scatter(train_size, R2_score, color='black', label='R2 Score')
    plt.xlabel('Training Data[0-1]')
    plt.ylabel('R2-Score')
    plt.title('Alpha=%1.2f' %a)

    plt.legend()
    i=i+1
    
    
    
    
    ######################Ridge Regression with cross validation##############################
the_grid=GridSpec.GridSpec(2,4)

fig=plt.figure(figsize=(15,10))
train_size=[.99, .9, .8, .7]
i=0
for x in train_size:
    train_error=[]
    test_error=[]
    error=[]
    valid_error=[]
    R2_score=[]
    alpha=[0.0,0.0001,0.001,0.01,0.1,1,1.5,2,3,4,5] 
    for a in alpha:
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = model_selection.train_test_split(X, y, test_size=1-x,train_size = x,shuffle=True)
#        X_mean_train=np.mean(diabetes_X_train, axis=0)
##        print(X_mean_train)
#        X_std_train=np.std(diabetes_X_train, axis=0)
##        print(X_std_train)
#        diabetes_X_train=(diabetes_X_train-X_mean_train)/X_std_train
#        
#        X_mean_test=np.mean(diabetes_X_test, axis=0)
#        X_std_test=np.std(diabetes_X_test, axis=0)
#        diabetes_X_test=(diabetes_X_test - X_mean_test)/X_std_test
#
        X_min_train=np.min(diabetes_X_train, axis=0)
        X_max_train=np.max(diabetes_X_train, axis=0)
        diabetes_X_train=np.abs(diabetes_X_train-X_min_train)/(X_max_train - X_min_train)
        
        
        X_min_test=np.min(diabetes_X_test, axis=0)
        X_max_test=np.max(diabetes_X_test, axis=0)
        diabetes_X_test=np.abs(diabetes_X_test - X_min_test)/(X_max_test-X_min_test)

        
        
        X_T=np.transpose(diabetes_X_train)
        temp=np.matmul(X_T,diabetes_X_train)
        temp=temp+a*np.eye(temp.shape[0], dtype=int)
        a1=np.linalg.inv(temp)
        b=np.matmul(X_T, diabetes_y_train)
        w=np.matmul(a1,b)
        diabetes_y_train_pred=np.matmul(diabetes_X_train,w)
        diabetes_y_test_pred=np.matmul(diabetes_X_test,w)
        y_train_mean=np.mean(diabetes_y_train)
        
        
#        y_tain_std=np.std(diabetes_y_train_pred)
#        y_test_mean=np.mean(diabetes_y_test_pred)    # diabetes_y_test_pred
#        y_test_std=np.std(diabetes_y_test_pred)
#        
#        diabetes_y_train_pred=(diabetes_y_train_pred*y_tain_std)+y_train_mean
#        diabetes_y_test_pred=(diabetes_y_test_pred*y_test_std)+y_test_mean
#      
        
        tot_var=np.sum((diabetes_y_train - y_train_mean)**2)/len(diabetes_y_train)
        error=np.sum((diabetes_y_train - diabetes_y_train_pred)**2)/len(diabetes_y_train)
        train_error.append(error)
        test_error.append(np.sum((diabetes_y_test - diabetes_y_test_pred)**2)/len(diabetes_y_test))
        R2_score.append(1-(error/tot_var))
        kf=KFold(n=len(diabetes_X_train), n_folds=5, shuffle=True)
        mse=0
        for train_indices, test_indices in kf:
            X_T=np.transpose(diabetes_X_train[train_indices])
            temp=np.matmul(X_T, diabetes_X_train[train_indices])
            temp=temp+a*np.eye(temp.shape[0], dtype=int)
            a=np.linalg.inv(temp)
            b=np.matmul(X_T, diabetes_y_train[train_indices] )
            w=np.matmul(a,b)
            X_train_predict=np.matmul(diabetes_X_train[test_indices], w)
            mse=mse+ np.sum((diabetes_y_train[test_indices] - X_train_predict)**2)/len(X_train_predict)
        valid_error.append(np.mean(mse)/5)
    plt.subplot(the_grid[0,i])
    plt.ylim((0,8000))
    plt.suptitle('Que 4) Training Error, Test Error, validation Error and R2 Score wrt Alpha on diabetes Without inbuilt function with normaliztion', fontsize=15)
    plt.plot(alpha, train_error , color='blue',marker='.', label='Training Error')
    plt.plot(alpha, test_error, color='red', marker='.',label='Test Error')
    plt.plot(alpha, valid_error, 'r--' ,marker='.', linewidth=2, label='Validation Error')
    
    plt.ylabel('Error')
    plt.xlabel('Alpha')
    plt.title('Training Data=%1.2f' %x)
    plt.legend()    
   
    plt.subplot(the_grid[1,i])
    plt.ylim((0,1))
    plt.scatter(alpha, R2_score, color='black', label='R2 Score')
    plt.xlabel('Alpha')
    plt.ylabel('R2-Score')
    plt.title('Training Data=%1.2f' %x)

    plt.legend()
    i=i+1
    
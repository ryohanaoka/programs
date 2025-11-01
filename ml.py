
from cmath import nan
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.linear_model import LogisticRegression

import pandas as pd
from numpy import hstack
from numpy import vstack
from numpy import asarray
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from statistics import variance as var
import math
import warnings

import threading
import time

warnings.simplefilter('ignore')
warnings.simplefilter('error',RuntimeWarning)

  
def SLearner(data,N,alpha,models):
    metamodel=LinearRegression(positive=True)
    Y=data.iloc[:,0]
    X=data.iloc[:,1:]

    rs=524
    
    meta_X=list()
    meta_Y=list()
    if N/10<10:
      kf=KFold(n_splits=int(N/10)+2,shuffle=True,random_state=rs)
    else:
      kf=KFold(n_splits=10,shuffle=True,random_state=rs)
    
    for train,test in kf.split(data):
        fold_yhats=list()
        trainY,trainX=data.iloc[train,0],data.iloc[train,1:]
        testY,testX  =data.iloc[test,0],data.iloc[test,1:]
        train,test   =data.iloc[train,:],data.iloc[test,:]
        for model in models: 
            model.fit(trainX, trainY)
            yhat = model.predict(testX)
            fold_yhats.append(yhat.reshape(len(yhat),1))

            
        meta_X.append(hstack(fold_yhats))
        meta_Y.extend(testY)

    meta_X=pd.DataFrame(vstack(meta_X))
    meta_Y=asarray(meta_Y)
    
    
    metamodel.fit(meta_X,meta_Y)
    
    newX1=pd.DataFrame(data.iloc[:,1:])
    newX0=pd.DataFrame(data.iloc[:,1:])
    
    newX1['x']=1
    newX0['x']=0
    
    newX=pd.concat([newX1,newX0])
    
    yhats=list()
    
    for model in models: 
        model.fit(X,Y)
        yhat=model.predict(newX)
        yhats.append(yhat.reshape(len(yhat),1))

    yhats=pd.DataFrame(hstack(yhats))
    
    pred=metamodel.predict(yhats)

    SL=np.mean(pred[:N])-np.mean(pred[-N:])

    coef=metamodel.coef_
    intcpt=metamodel.intercept_

    return SL

def TLearner(data,N,alpha,models):#including T-, X-, DR-learner
    metamodel1=LinearRegression(positive=True)
    metamodel0=LinearRegression(positive=True)

    rs=524

    Y_all=data.iloc[:,0]
    X_all=data.iloc[:,1:]

    data1_all=data[data['x'] == 1]
    data0_all=data[data['x'] == 0]
    
    Y1_all=data1_all.iloc[:,0]
    X1_all=data1_all.iloc[:,2:]

    Y0_all=data0_all.iloc[:,0]
    X0_all=data0_all.iloc[:,2:]

    Y=data.iloc[:,0]
    X=data.iloc[:,1:]
    
    data1=data[data['x'] == 1]
    data0=data[data['x'] == 0]
    
    Y1=data1.iloc[:,0]
    X1=data1.iloc[:,2:]
    
    Y0=data0.iloc[:,0]
    X0=data0.iloc[:,2:]

    
    meta_X1=list()
    meta_X0=list()
    meta_Y1=list()
    meta_Y0=list()



    flg=1
    ret=0

    while flg!=0:
        flg=0
        ret += 1
        rs += ret
        if len(Y0)/10<10:
            kf0=KFold(n_splits=int(len(Y0)/10)+3,shuffle=True,random_state=rs)
        else:
            kf0=KFold(n_splits=10,shuffle=True,random_state=rs)

        if len(Y1)/10<10:
            kf1=KFold(n_splits=int(len(Y1)/10)+3,shuffle=True,random_state=rs+1)
        else:
            kf1=KFold(n_splits=10,shuffle=True,random_state=rs+1)

        for n_train,n_test in kf1.split(data1):
            train,test   =data1.iloc[n_train,:],data1.iloc[n_test,:]
            if len(train)<=1 or len(test)<=1:
                flg += 1

        for n_train,n_test in kf0.split(data0):
            train,test   =data0.iloc[n_train,:],data0.iloc[n_test,:]
            if len(train)<=1 or len(test)<=1:
                flg += 1

        # print(ret)

    
    for n_train,n_test in kf1.split(data1):
        fold_yhats=list()
        trainY,trainX=data1.iloc[n_train,0],data1.iloc[n_train,1:]
        testY,testX  =data1.iloc[n_test,0],data1.iloc[n_test,1:]
        for model in models: 
            model.fit(trainX, trainY)
            yhat = model.predict(testX)
            fold_yhats.append(yhat.reshape(len(yhat),1))

            
        meta_X1.append(hstack(fold_yhats))
        meta_Y1.extend(testY)

    meta_X1=pd.DataFrame(vstack(meta_X1))
    meta_Y1=asarray(meta_Y1)
    

    for n_train,n_test in kf0.split(data0):
        fold_yhats=list()
        trainY,trainX=data0.iloc[n_train,0],data0.iloc[n_train,1:]
        testY,testX  =data0.iloc[n_test,0],data0.iloc[n_test,1:]
        for model in models: 
            model.fit(trainX, trainY)
            yhat = model.predict(testX)
            fold_yhats.append(yhat.reshape(len(yhat),1))

            
        meta_X0.append(hstack(fold_yhats))
        meta_Y0.extend(testY)

    meta_X0=pd.DataFrame(vstack(meta_X0))
    meta_Y0=asarray(meta_Y0)

    if np.isnan(meta_X1).any().any() | np.isnan(meta_Y1).any() | np.isnan(meta_X0).any().any() | np.isnan(meta_Y0).any():

        return [None]*3
    else:
        try:


            metamodel1.fit(meta_X1,meta_Y1)
            # print(metamodel1.summary())

            metamodel0.fit(meta_X0,meta_Y0)
            # print(metamodel0.summary())
        
            newX=data.iloc[:,2:]
        
            yhats1=list()
            yhats1_=list()
            yhats0=list()
            yhats0_=list()

            for model in models: 
                model.fit(X1,Y1)
                yhat1=model.predict(newX)
                yhats1.append(yhat1.reshape(len(yhat1),1))
                yhat1_=model.predict(X0)
                yhats1_.append(yhat1_.reshape(len(yhat1_),1))
            
                model.fit(X0,Y0)
                yhat0=model.predict(newX)
                yhats0.append(yhat0.reshape(len(yhat0),1))
                yhat0_=model.predict(X1)
                yhats0_.append(yhat0_.reshape(len(yhat0_),1))
            
            yhats1=pd.DataFrame(hstack(yhats1))
            yhats0=pd.DataFrame(hstack(yhats0))

            yhats1_=pd.DataFrame(hstack(yhats1_))
            yhats0_=pd.DataFrame(hstack(yhats0_))

            pred1=metamodel1.predict(yhats1)
            pred0=metamodel0.predict(yhats0)
        
            TL=np.mean(pred1)-np.mean(pred0)

            #XL
            predY0=metamodel0.predict(yhats0_)
            predY1=metamodel1.predict(yhats1_)

            pred1=Y1-predY0
            pred0=predY1-Y0

            mu_1 = RandomForestRegressor(random_state=rs)
            mu_1.fit(X1, pred1)

            mu_0 = RandomForestRegressor(random_state=rs)
            mu_0.fit(X0, pred0)

            tau_1=mu_1.predict(X_all.iloc[:,1:])
            tau_0=mu_0.predict(X_all.iloc[:,1:])

            PS = LogisticRegression()
            PS.fit(X_all.iloc[:,1:],X_all.iloc[:,0])
        
            PS_pred=PS.predict_proba(X_all.iloc[:,1:])[:,1]

            tau=PS_pred*(tau_1)+(1-PS_pred)*(tau_0)
        
            XL=np.mean(tau)

            #DRL

            pred1=metamodel1.predict(yhats1)
            pred0=metamodel0.predict(yhats0)

            PS.fit(X.iloc[:,1:],X.iloc[:,0])
            PS_pred=PS.predict_proba(X.iloc[:,1:])[:,1]
            T=X.iloc[:,0]
            tau_AIPW=(pred1+T*(Y-pred1)/PS_pred)-(pred0+(1-T)*(Y-pred0)/(1-PS_pred))

            tau_DRL= RandomForestRegressor(random_state=rs)
            tau_DRL.fit(X.iloc[:,1:], tau_AIPW)

            DRL_HTE=tau_DRL.predict(X_all.iloc[:,1:])

            DRL=np.mean(DRL_HTE)

            out=[TL,XL,DRL]
        except ValueError:
            out=[None,None,None]
        except RuntimeWarning:
            out=[None,None,None]
            #print("error")
        except RuntimeError:
            out=[None,None,None]
        finally:
            return out




def get_models():
    models = list()
    models.append(LinearRegression())
    models.append(Lasso())
    models.append(Ridge())
    models.append(ElasticNet())
    models.append(Lars())
    models.append(LassoLars())
    models.append(OrthogonalMatchingPursuit())
    models.append(BayesianRidge())
    models.append(PassiveAggressiveRegressor())
    models.append(RANSACRegressor(min_samples=2))
    models.append(TheilSenRegressor())
    models.append(HuberRegressor())
    models.append(KernelRidge())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(RandomForestRegressor(n_estimators=10))
    models.append(ExtraTreesRegressor(n_estimators=10))
    models.append(AdaBoostRegressor())
    models.append(GradientBoostingRegressor())
    models.append(MLPRegressor())
    models.append(XGBRegressor())
    
    return models
  


#for simple test
def get_models2():
    models = list()
    models.append(LinearRegression())
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(RandomForestRegressor(n_estimators=10))
    models.append(XGBRegressor())
    return models


def get_models3():
    models = list()
    models.append(LinearRegression())
    return models

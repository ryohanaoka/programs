import numpy as np
import pandas as pd
import random
from scipy.special import expit
import csv
import os


def makec(N):
    c=pd.DataFrame()
    for i in range(10):
     c_=np.random.normal(0,1,N)
     c=pd.concat([c,pd.DataFrame(c_-c_.mean())],axis=1)
    return pd.DataFrame(c)


def makex(N):
  x=random.choices([0,1], k=N)
  return pd.DataFrame(x)

def makealpha(c):
    tmp=(1/(1+np.exp(-12*(c-0.5))-1/2))
    alpha=1+4/10*tmp.sum(axis=1)

    return alpha


def makeY(c,x,N,alpha,sen,tau):
  beta0_=np.array([1,1,2,2,4,4,8,8,16,16])
  beta1_=np.array([1,1,1,1,1,1,1,1,1,1])
  beta3=np.array([1,np.sqrt(2),np.sqrt(3),np.sqrt(2),1])*0.5
  beta4=np.array([2,4,6,8,10])
  
  var=2

  b0=np.sqrt(1/(sum(beta0_^2)/var))
  beta0=b0*beta0_
  
  if sen==("a" or "b" or "c"):
    b1=0
  else:
      b1=0.75

  beta1=b1*beta1_
 
  if(tau=="fc"):
    sgm=abs(1+c.iloc[:,0]*np.random.rand())
  else:
      sgm=np.repeat(np.sqrt(var),N)

  cint=pd.DataFrame()
  for i in range(9):
    cint=pd.concat([cint,c.iloc[:,i]*c.iloc[:,i+1]],axis=1)

  gma=np.array([1,1,1,1,1,1,1,1,1])*np.sqrt(var/10)
  
  Y=[]

  if sen=="a" or sen=="d":
      for i in range(N):
        Y=np.append(Y,np.random.normal(x.iloc[i,0]*alpha[i]+np.dot(beta0,c.iloc[i,:])+
                                   x.iloc[i,0]*np.dot(beta1,c.iloc[i,:]),sgm[i],1))
  elif sen=="b" or sen=="e":
      for i in range(N):
        Y=np.append(Y,np.random.normal(x.iloc[i,0]*alpha[i]+np.dot(beta0,c.iloc[i,:])+
                                   np.dot(gma,cint.iloc[i,:])+
                                   x.iloc[i,0]*np.dot(beta1,c.iloc[i,:]),sgm[i],1))
  else:
      for i in range(N):
          Y=np.append(Y,np.random.normal(x.iloc[i,0]*alpha[i]+
                                         np.dot(beta3,c.iloc[i,:5])*np.dot(beta3,c.iloc[i,5:])+
                                         0.5*expit(np.dot(beta4,c.iloc[i,[0,2,4,6,8]]))+
                                         x.iloc[i,0]*np.dot(beta1,c.iloc[i,:]),sgm[i],1))

  
  
  return pd.DataFrame(Y)


def makedata(N,alpha,sen,tau):
    
    c=makec(N)
    x=makex(N)
    if tau=="fc":
        alpha=makealpha(c)
    else:
        alpha=pd.Series(np.repeat(alpha,N))
    Y=makeY(c,x,N,alpha,sen,tau)
    

    data=pd.concat([Y,x,c,alpha],axis=1)

    data=data.set_axis(["Y","x","c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","alpha"],axis=1)

    return data
  

if __name__=="__main__":

  random.seed(524)
  filename=os.path.dirname(__file__)+"\yvars.csv"
  with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
      Yvars=row

  taus=["0","s","fc"]
  sens=["a","b","c","d","e","f"]
  Ns=[50,100,200,500]
  
  for sen in sens:
    ALPHA=np.sqrt(float(Yvars[(2*sens.index(sen)+1)%6]))
    alphas=[0,ALPHA,0]
    for k in range(3):
      tau=taus[k]
      alpha=alphas[k]
      for i in range(500):
        for N in Ns:
          filename=os.path.dirname(__file__)+"\\data\\"+sen+"\\"+str(N)+"\\data_sen_"+sen+"_"+tau+"_"+str(N)+"_"+str(i)+"_.csv"
          out=makedata(N,alpha,sens.index(sen),tau)
          
          #out.to_csv(filename,index=False)


            
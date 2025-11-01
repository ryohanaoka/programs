#install.packages("speedlm")
#install.packages("speedglm")
#install.packages("kernlab")
#install.packages("xgboost")
#install.packages("earth")
#install.packages("e1071")
#install.packages("polspline")
#install.packages("arm")

library(sandwich)
library(lmtest)
library(PSweight)
library(SuperLearner)
library(AIPW)
library(ranger)
library(glmnet)
library(doParallel)

#ANCOVA1
ancova1<-function(data,alpha){
  model<-lm(Y~.,data=data)

  est<-summary(model)$coef[2,1]
  
  LCI<-coefci(model,vcov=sandwich)[2,1]
  UCI<-coefci(model,vcov=sandwich)[2,2]
  coverage<- LCI<=mean(alpha[,1]) && mean(alpha[,1]) <=UCI
  
  estvar<-coeftest(model,vcov=sandwich)[2,2]^2
  return(cbind(est,coverage,estvar))
}


#ANCOVA2
ancova2<-function(data,alpha){
  model<-lm(Y~x+c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+x*(c1+c2+c3+c4+c5+c6+c7+c8+c9+c10),data=data)
  
  est<-summary(model)$coef[2,1]
 
  LCI<-coefci(model,vcov=sandwich)[2,1]
  UCI<-coefci(model,vcov=sandwich)[2,2]
  coverage<- LCI<= mean(alpha[,1]) && mean(alpha[,1]) <=UCI
  
  estvar<-coeftest(model,vcov=sandwich)[2,2]^2
  return(cbind(est,coverage,estvar))
}


#AIPW
aipw<-function(data,alpha){
  ル
  aipw<-AIPW$new(Y=data$Y,A=data$x,W=data[,3:12],Q.SL.library=c("SL.speedlm"),g.SL.library=c("SL.speedglm"))

  
  suppressMessages(aipw$fit())
  aipw$summary()

  est<-aipw$estimates$RD[1]

  coverage<-aipw$estimates$RD[3]<=mean(alpha[,1]) && mean(alpha[,1])<=aipw$estimates$RD[4]

  estvar<-aipw$estimates$RD[2]^2
  
  return(cbind(est,coverage,estvar))
} 


#AIPW（ML）
aipw2<-function(data,alpha,models){
  models<-c("SL.lm","SL.glmnet","SL.ridge","SL.bayesglm","SL.earth","SL.ksvm","SL.nnet",
            "SL.ranger","SL.rpart","SL.stepAIC","SL.svm","SL.xgboost","SL.polymars")
  
  
  aipw<-AIPW$new(Y=data$Y,A=data$x,W=data[,3:12],Q.SL.library=models,
                 g.SL.library=c("SL.speedglm"))
  
  suppressMessages(aipw$fit())
  aipw$summary()
  
  est<-aipw$estimates$RD[1]

  coverage<-aipw$estimates$RD[3]<=mean(alpha[,1]) && mean(alpha[,1])<=aipw$estimates$RD[4]
  #分散推定値
  estvar<-aipw$estimates$RD[2]**2
  
  return(cbind(est,coverage,estvar))
} 



sim<-function(sen,N,tau,i){
  filename<-paste0(getwd(),"/data/",as.character(sen),"/",as.character(N),"/data_sen_",sen,"_",tau,"_",as.character(N),"_",as.character(i),"_.csv")
  DATA<-read.csv(filename)
  
  data<-DATA[1:12]

  alpha<-DATA[13]
  
  
  anc1<-ancova1(data,alpha)
  anc2<-ancova2(data,alpha)
  AIPW1<-aipw(data,alpha)
  AIPW2<-aipw2(data,alpha,models)
  
  return(cbind(anc1,anc2,AIPW1,AIPW2))
}


PACK<-c("sandwich","lmtest","AIPW","SuperLearner")

ncore<-getOption("mc.cores",detectCores())-2




foreach(sen=c("a","e"))%do%{
  foreach(N=c(50,100,200,500))%do%{
    foreach(tau=c("0","fc"))%do%{
      cl <- makePSOCKcluster(ncore)
      registerDoParallel(cl)
      result<-foreach(i=0:499,.combine = "rbind",.packages=PACK)%dopar%{
        sim(sen,N,tau,i)
      }
      stopCluster(cl)

      outfilename<-paste0(getwd(),"/out/",as.character(sen),"/out_sen_",sen,"_",tau,"_",as.character(N),"_.csv")

      write.csv(result,outfilename,row.names=FALSE)
    }
  }
}


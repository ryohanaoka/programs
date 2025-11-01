import numpy as np
import pandas as pd
import random
import ml
import math
from statistics import variance as var
import time
import os
import csv
from scipy import stats
import datetime

def boot(data,N,ALPHA,B,models,sen,tau,i,start):
    estSL=list()
    estTL=list()
    estXL=list()
    estDRL=list()

    coverage_pt=list()
    coverage_t=list()

    LCI_pt=list()
    UCI_pt=list()
    LCI_t=list()
    UCI_t=list()
    estvar=list()

    tau_true=np.mean(ALPHA)

    methods=["SL","TL","XL","DRL"]

    random.seed(524) 


    while len(estSL)<B:
        #start_time=time.time()
        path = os.path.dirname(__file__)+"/../out/"+sen+"/"+tau+"/"+str(N)+"/progress/prog_out_sen_"+sen+"_"+tau+"_"+str(N)+"_"+str(start)+".txt"
        if i%10==0:
            if len(estSL)%10==0:
                f = open(path, 'a', encoding='UTF-8')
                f.write("N:"+str(N)+",i:"+str(i)+"B:"+str(len(estSL))+":"+str(datetime.datetime.now())+"\n")
                f.close()


        data1=data[data['x'] == 1]
        data0=data[data['x'] == 0]

        bootsample1=data1.sample(n=len(data1),replace=True)
        bootsample0=data0.sample(n=len(data0),replace=True)
        bootsample=pd.concat([bootsample1,bootsample0])
    
        

        tmp=ml.TLearner(bootsample,N,ALPHA,models)

        if not tmp[0]:
            print("error")
        else:        
            estTL.append(tmp[0])
            estXL.append(tmp[1])
            estDRL.append(tmp[2])

            estSL.append(ml.SLearner(bootsample,N,ALPHA,models))

        ests= [estSL,estTL,estXL,estDRL]
    for est in ests:
        est.sort()

        #parcentile
        tmp_LCI_pt=(est[math.floor((B-1)*(1-0.95))]+est[math.ceil((B-1)*(1-0.95))])/2
        tmp_UCI_pt=(est[math.floor((B-1)*0.95)]+est[math.ceil((B-1)*0.95)])/2

        LCI_pt.append(tmp_LCI_pt)
        UCI_pt.append(tmp_UCI_pt)

        coverage_pt.append(int((tmp_LCI_pt<=tau_true) & (tau_true<=tmp_UCI_pt)))
        
        
        
        
        tmp_t=stats.norm.interval(0.95,
                             loc=np.mean(est),
                             scale=stats.tstd(est))
        
        #t-dist
        LCI_t.append(tmp_t[0])
        UCI_t.append(tmp_t[1])

        coverage_t.append(int(((tmp_t[0])<=tau_true) & (tau_true<=tmp_t[1])))
        
        try:
            estvar.append(float(var(est)))
        except OverflowError:
            met=methods[ests.index(est)]
            outfilename=os.path.dirname(__file__)+"/../out/"+sen+"/"+tau+"/"+str(N)+"/ex/"+sen+"/ex_out_sen_"+sen+"_"+tau+"_"+str(N)+"_"+met+"_"+str(i)+".csv"
            with open(outfilename, 'w', encoding='UTF-8') as f:
                writer=csv.writer(f)
                writer.writerow(est)
            estvar.append(99999.99999)
        
        #print(estvar)

    estvar.extend(LCI_pt)
    estvar.extend(UCI_pt)
    estvar.extend(coverage_pt)
    estvar.extend(LCI_t)
    estvar.extend(UCI_t)
    estvar.extend(coverage_t)

    print(estvar)

    return estvar
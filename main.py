# %%
import ml
import boot
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import time
import os
import csv
import pathlib
import datetime
import sys

# %%
def op(start,length ,N, alpha,sen,tau):
    result = []
    for idx in range(length):
        b = (N, alpha, sen,tau,start+idx+1,start)
        result.append(b)
    return result



def sim(args_tuple):
    N, alpha,sen,tau, i,start = args_tuple

    DATA=pd.read_csv(os.path.dirname(__file__)+"/../run_data/"+sen+"/"+tau+"/"+str(N)+"/data_sen_"+sen+"_"+tau+"_"+str(N)+"_"+str(i)+"_.csv")

    data=DATA.drop("alpha",axis=1)    
    
    alpha=DATA["alpha"]
    #print(alpha)
      
    models=ml.get_models()

    data1=data[data['x'] == 1]
    data0=data[data['x'] == 0]

    UNADJ=np.mean(data1['Y'])-np.mean(data0['Y'])

    SL=ml.SLearner(data,N,alpha,models)
    TL=ml.TLearner(data,N,alpha,models)
    
    B=500

    varcov=boot.boot(data,N,alpha,B,models,sen,tau,i,start)

    tau_true=np.mean(alpha)

    out=[tau_true,UNADJ]
    out.extend(SL)
    out.extend(TL)
    out.extend(varcov)


    return out

#%%


    

if __name__ == "__main__":
    args=sys.argv

    it=500

    sen=args[1]

    tau=args[2]

    N=int(args[3])

    start=int(args[4])

    taus=["0","fc"]

    sens=["a","e"]
    
    k=taus.index(tau)

    alpha=0
    
    start_time=time.time()
    results=[]
    args=op(start,it,N,alpha,sen,tau)
                
    outfilename=os.path.dirname(__file__)+"/../out/"+sen+"/"+tau+"/"+str(N)+"/out_sen_"+sen+"_"+tau+"_"+str(N)+"_"+str(start)+".csv"
                
    with ProcessPoolExecutor(max_workers = 50) as e:
        results=e.map(sim,args)


                    
    with open(outfilename,"w") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerows(results)
        
            
    end_time=time.time()
    print(end_time-start_time)




#%%


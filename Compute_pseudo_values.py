import numpy as np
import pandas as pd

def compute_d_and_n(data, t):
    temp = data[data["time"] == t].groupby("status").count()
    try:
        d = temp.loc[1.0, "time"]
    except KeyError:
        d = 0
    try:
        c = temp.loc[0.0, "time"] 
    except KeyError:
        c = 0 
    return d, c


def compute_d_and_n_matrix(data):
    timeline = data["time"].sort_values().unique()
    di = np.full(len(timeline) + 1, np.nan)
    ci = np.full(len(timeline) + 1, np.nan)
    ni = np.full(len(timeline) + 1, 0)
    ni[0] = data.shape[0]
    for i in range(len(timeline)):
        d, c = compute_d_and_n(data, timeline[i])
        di[i] = d
        ci[i] = c
    m = pd.DataFrame(index=timeline)
    m["ni"] = ni[:-1]
    m["di"] = di[:-1]
    m["ci"] = ci[:-1]
    return m


def survival_function(time, status):
    Surv_Eval_Time=sorted(np.unique(time))
    df=pd.concat([pd.DataFrame(time), pd.DataFrame(status)], axis=1)
    df.columns=['time','status']
    partial_matrix=compute_d_and_n_matrix(df)
    partial_matrix.reset_index(inplace=True)
    partial_matrix = partial_matrix.rename(columns = {'index':'time'})
    partial_matrix.columns=['time','nrisk','di','ci']
    full_matrix=np.array(partial_matrix)
    r_last = full_matrix[0, 1]
    d_last = full_matrix[0, 2]
    c_last = full_matrix[0, 3]
    for j in range(1, full_matrix.shape[0]):
        r = r_last - d_last - c_last
        d = full_matrix[j, 2]
        c = full_matrix[j, 3]
        if r < 0:
            full_matrix[j, 1] = 0
        else:
            full_matrix[j, 1] = r
        if d < 0:
            full_matrix[j, 2] = 0
        if c < 0:
            full_matrix[j, 3] = 0        
        r_last = r
        d_last = d
        c_last = c
    surv=np.cumprod(1-(full_matrix[:,2]/full_matrix[:,1])) 
    
    
    return partial_matrix, surv


def loo_survival_function(partial_matrix, time, status):
    Surv_Eval_Time=np.array(partial_matrix.loc[:,'time'])
    loo_surv_prob=np.zeros((len(time),len(Surv_Eval_Time)))
    loo_matrix=np.zeros((len(time),len(Surv_Eval_Time),3))
    for i in range(len(time)):
        loo_matrix[i,0,0]=-1
        idx=np.where(time[i]==Surv_Eval_Time)
        if status[i]==1:
            loo_matrix[i,idx,1]=-1
        elif status[i]==0:
            loo_matrix[i,idx,2]=-1 
        loo_matrix[i,:,:]=np.array(partial_matrix)[:,1:]+loo_matrix[i,:,:]
        for p in range(len(Surv_Eval_Time)-1):
            loo_matrix[i,(p+1),0]=loo_matrix[i, (p),0]-(loo_matrix[i, (p),1]+loo_matrix[i,(p),2])   
    for j in range(len(time)):        
        loo_surv_prob[j,:]=np.cumprod(1-(loo_matrix[j,:,1]/loo_matrix[j,:,0]))
    return loo_surv_prob

def pseudo_values(data, evaltime):
    """Calculate the pseudo values for Survival function.
    Arguments:
        time: Array of survival time 
        status: Array of event status
        evaltime: Prespecified time points at which pseudo values are calcuated
    Returns:
        A dataframe of pseudo values for survival function for all subjects in the data at the prespecified time points. 
    """
    time=np.array(data['time'])
    status=np.array(data['status'])
    surv=survival_function(time, status)
    loo_surv=loo_survival_function(surv[0], time, status)
    pseudo_values= surv[0].iloc[0,1]*surv[1]-((surv[0].iloc[0,1]-1)*loo_surv)

    index=np.zeros((len(evaltime)))
    for t in range(len(evaltime)):
        index[t]=np.where((np.array(surv[0])[:,0]<=evaltime[t]))[0][-1]
    pseudo_values=np.array(pd.DataFrame(pseudo_values).iloc[:,index])    
    return pd.DataFrame(pseudo_values)


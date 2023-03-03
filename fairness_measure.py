import numpy as np
from sklearn.metrics.pairwise import cosine_distances
# =============================================================================
#                             Individual Fairness
# =============================================================================

def individual_fairness(prediction,X, ntimes, scale_fairness):
    """
    Compute average individual fairness measures
    Arguments:
    prediction -- Predicted survival probability.
    X -- Covariates 
    ntimes--Number of pre-specified time points
    scale_fairness--Scale parameter
    Returns:
    Average individual fairness measures
    """  

    N = len(prediction)
    R_beta = 0.0 #initialization of individual fairness
    for i in range(ntimes):
        for j in range(prediction.shape[0]):
            for k in range(prediction.shape[0]):
                if k<=j:
                    continue
                else:
#                     distance = np.sqrt(sum((X[j]-X[k])**2)) # Euclidean distance
                    distance=cosine_distances(np.array(X)[j].reshape(1,-1),np.array(X)[k].reshape(1,-1))[0][0] ##The Cosine distance between two individuals
                    R_beta = R_beta + max(0,(np.abs(prediction[j,i]-prediction[k,i])-(scale_fairness*distance)))   
    R_beta_avg = 2*R_beta/(N*(N-1)*ntimes)
    return R_beta_avg

# =============================================================================
#                             Censoring based Individual  Fairness
# =============================================================================

def censoring_individual_fairness(prediction_uncen, prediction_cen, X_distance_uncen, X_distance_cen, time_uncen, time_cen, ntimes,scale_fairness):

    """
    Compute average Censoring-based individual fairness measures
    Arguments:
    prediction_uncen -- Predicted survival probability for uncensored individuals.
    prediction_cen -- Predicted survival probability for censored individuals.
    X_distance_uncen -- Covariates of uncensored individuals. 
    X_distance_cen -- Covariates of censored individuals.
    time_uncen -- Observed time of uncensored individuals.
    time_cen -- Observed time of censored individuals.
    ntimes--Number of pre-specified time points
    scale_fairness--Scale parameter
    Returns:
    Average Censoring-based individual fairness measures
    """      
    
    N_uncen = len(prediction_uncen)
    N_cen = len(prediction_cen)

    R_beta=0.0 #initialization of censoring based individual fairness
    for i in range(ntimes):
        for j in range(N_cen):
            for k in range(N_uncen):
                if N_cen==0:
                    break
                elif time_cen[j]>time_uncen[k]:
                    continue
                else:
                    distance=cosine_distances(np.array(X_distance_cen)[j].reshape(1,-1),np.array(X_distance_uncen)[k].reshape(1,-1))[0][0]
                    R_beta = R_beta + max(0,(np.abs(prediction_cen[j,i]-prediction_uncen[k,i])-(scale_fairness*distance))) 
    if N_cen==0:
        R_beta_avg=0.0
    else:
        R_beta_avg = R_beta/(N_cen*N_uncen*ntimes)
    return R_beta_avg


# =============================================================================
#                             Group Fairness
# =============================================================================


def group_fairness(prediction,S,ntimes): 
    
    """
    Compute average group fairness measures
    Arguments:
    prediction -- Predicted survival probability.
    S -- Protected attribute 
    ntimes--Number of pre-specified time points
    Returns:
    Average group fairness measures
    """      
    group_fairness=np.zeros((ntimes))
    for i in range(ntimes):
        pred=prediction[:,i]
        unique_group = np.unique(S)
        avg_pred_ratio = sum(pred)/len(pred)
    
        numClasses = len(unique_group)
        concentrationParameter = 1.0
        dirichletAlpha = concentrationParameter/numClasses
    
        pred_ratio_group = np.zeros((len(unique_group)))
        group_total = np.zeros((len(unique_group)))
    
        for j in range(len(pred)):
            pred_ratio_group[S[j]-1] = pred_ratio_group[S[j]-1] + pred[j]
            group_total[S[j]-1] = group_total[S[j]-1] + 1  
    
        avg_pred_ratio_group = (pred_ratio_group+dirichletAlpha)/(group_total+concentrationParameter)
    
        group_fairness[i] = np.max(np.abs(avg_pred_ratio_group-avg_pred_ratio))
    return np.mean(group_fairness)



# =============================================================================
#                             Censoring Based Group Fairness
# =============================================================================
def censoring_group_fairness(prediction_uncen, prediction_cen, X_distance_uncen, X_distance_cen, time_uncen, time_cen, ntimes,scale_fairness, dataset_name):
    
    """
    Compute average Censoring-based group fairness measures
    Arguments:
    prediction_uncen -- Predicted survival probability for uncensored individuals.
    prediction_cen -- Predicted survival probability for censored individuals.
    X_distance_uncen -- Covariates of uncensored individuals. 
    X_distance_cen -- Covariates of censored individuals.
    time_uncen -- Observed time of uncensored individuals.
    time_cen -- Observed time of censored individuals.
    ntimes--Number of pre-specified time points
    scale_fairness--Scale parameter
    dataset_name -- Name of the dataset
    Returns:
    Average Censoring-based group fairness measures
    """     
    if dataset_name=='SUPPORT':
        protected_group=["race_1","race_2"] # Groups in the protected attribute
    elif dataset_name=='SEER':
        protected_group=["Race_ord_1","Race_ord_2","Race_ord_3","Race_ord_4"]        
    elif dataset_name=='FLChain':
        protected_group=["sex_1","sex_0"]   
    
    N_uncen = len(prediction_uncen)
    N_cen = len(prediction_cen)
    R_beta_total=0.0 #initialization of overall fairness
    for group in protected_group:
        uncen_idx=np.where(np.array(X_distance_uncen[group]==1))[0]
        cen_idx=np.where(np.array(X_distance_cen[group]==1))[0]

        N_uncen_group=len(uncen_idx)
        N_cen_group=len(cen_idx)

        R_beta=0.0  #initialization of censoring based individual fairness for a paricular group
        for i in range(ntimes):
            for j in range(N_cen_group):
                for k in range(N_uncen_group):
                    if N_cen_group==0:
                        break
                    elif time_cen[cen_idx[j]]>time_uncen[uncen_idx[k]]:
                        continue
                    else:
                        distance=cosine_distances(np.array(X_distance_cen)[cen_idx[j]].reshape(1,-1),np.array(X_distance_uncen)[uncen_idx[k]].reshape(1,-1))[0][0]
                        R_beta = R_beta + max(0,(np.abs(prediction_cen[cen_idx[j],i]-prediction_uncen[uncen_idx[k],i])-(scale_fairness*distance))) 
        R_beta_total += R_beta                
    if N_cen==0:
        R_beta_avg=0.0
    else:        
        R_beta_avg = (R_beta_total)/(N_cen*N_uncen*ntimes)
    return R_beta_avg


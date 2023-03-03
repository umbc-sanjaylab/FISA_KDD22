import numpy as np
import pandas as pd
from random import sample
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from Compute_pseudo_values import *
np.random.seed(1234)



# ============================================================================================
#                             Convert the categorical variables into dummy variables
# ============================================================================================

def get_dummies(data, encode):
    """
    Convert categorical variable into dummy/indicator variables.
    Arguments:
    data -- Data of which to get dummy indicators.
    encode -- categorical columns   
    Returns:
    Dummy-coded data
    """    
    data_encoded = data.copy()
    encoded = pd.get_dummies(data_encoded, prefix=encode, columns=encode) ## Get dummies of the categorical variables
    return encoded

# =====================================================================================================
#                             Pre-specified time points for pseudo values calculation and evaluation
# =====================================================================================================

def Evaltime(tr_time):
    """
    Compute prespecified time points at which pseudo values will be calculated.
    Arguments:
    tr_time -- Observed times in training data. 
    Returns:
    prespecified time points at which pseudo values will be calculated
    """      
    eval_time=np.ceil(np.quantile(tr_time, [0.25,0.5,0.75]))
    return eval_time



# ===================================================================================
#                             Data Preprocessing
# ===================================================================================


def data_preprocess(data_csv_file, dataset_name):
    
    """Preprocess the data.
    Arguments:
        data_csv_file: Dataset in csv format. load the csv file from data directory.
        dataset_name: The name of the dataset 
    Returns:
        All the attributes needed for modeling, prediction and evaluation
    """
    data_df=pd.read_csv(data_csv_file)
    if dataset_name=='SUPPORT':
        one_hot_encoder_list = ['sex','race','dzgroup','dzclass','num.co', 'ca','adlp']  ## List of categorical columns
        standardized_list = ['age', 'slos', 'edu', 'scoma','avtisst',  'hday', 'meanbp','wblc', 'hrt', 'resp',
                         'temp', 'pafi', 'alb', 'bili', 'crea', 'sod','ph', 'glucose', 'bun', 'urine', 
                         'adlsc', 'sps', 'aps']  ## List of continuous columns which will be standardized
        protected_group1='age'  ## One of the protected attributes
        protected_group2=["race_1","race_2"] ## Groups in one of the protected attributes
    elif dataset_name=='SEER':
        data_df=data_df.loc[data_df['Race_ord']!=5]
        one_hot_encoder_list = ['Race_ord', 'Marital_3', 'Area_3', 'Histology_ord',
       'Grade_3', 'Tumor_size_ord', 'T.stage_ord', 'RT_ord', 'Surg_ord']
        standardized_list = ['Ageatdiagnosis']
        protected_group1='Ageatdiagnosis'
        protected_group2=["Race_ord_1","Race_ord_2","Race_ord_3","Race_ord_4"]        
    elif dataset_name=='FLChain':
        one_hot_encoder_list = ['sex', 'sample.yr','flc.grp']
        standardized_list = ['age', 'kappa', 'lambda', 'creatinine']
        protected_group1='age'
        protected_group2=["sex_1","sex_0"]         
        
    dataset = get_dummies(data_df, encode=one_hot_encoder_list)  ##Convert the categorical variables into dummy variables.
    data={}
    uncen_data={}
    uncen_train_val_data={}
    uncen_train_data={}
    uncen_val_data={}
    uncen_test_data={}
    cen_data={}
    cen_train_val_data={}
    cen_train_data={}
    cen_val_data={}
    cen_test_data={}
    for i in protected_group2:    
        data[i]=dataset[dataset[i]==1] ## Extract the data for the particular protected group
        uncen_data[i]=data[i][data[i]["status"]==1] ## Extract the uncensored data from the particular protected group's data
        cen_data[i]=data[i][data[i]["status"]==0]  ## Extract the censored data from the particular protected group's data
        uncen_train_val_data[i]=uncen_data[i].iloc[:int(0.8*len(uncen_data[i])),:] ## Split uncensored data into training (80%) & test (20%) set
        uncen_test_data[i]=uncen_data[i].iloc[int(0.8*len(uncen_data[i])):,:]
        uncen_train_data[i]=uncen_train_val_data[i].iloc[:int(0.8*len(uncen_train_val_data[i])),:] ## Split uncensored training data again into train (80%) & validation (20%) set
        uncen_val_data[i]=uncen_train_val_data[i].iloc[int(0.8*len(uncen_train_val_data[i])):,:]   
        cen_train_val_data[i]=cen_data[i].iloc[:int(0.8*len(cen_data[i])),:] ## Split censored data into training (80%) & test (20%) set
        cen_test_data[i]=cen_data[i].iloc[int(0.8*len(cen_data[i])):,:]
        cen_train_data[i]=cen_train_val_data[i].iloc[:int(0.8*len(cen_train_val_data[i])),:] ## Split censored training data again into train (80%) & validation (20%) set
        cen_val_data[i]=cen_train_val_data[i].iloc[int(0.8*len(cen_train_val_data[i])):,:]

    tr_df=[]
    va_df=[]
    te_df=[]
    for i in protected_group2:    
        tr_df.append(pd.concat([uncen_train_data[i],cen_train_data[i]])) ## Append censored and uncensored training observations from each group of the protected attribute
        va_df.append(pd.concat([uncen_val_data[i],cen_val_data[i]])) ## Append censored and uncensored validation observations from each group of the protected attribute
        te_df.append(pd.concat([uncen_test_data[i],cen_test_data[i]])) ## Append censored and uncensored test observations from each group of the protected attribute

    tr_data=pd.concat(tr_df).sample(frac =1.0).reset_index(drop=True) #Suffle the dataset for randomization
    va_data=pd.concat(va_df).sample(frac =1.0).reset_index(drop=True)
    te_data=pd.concat(te_df).sample(frac =1.0).reset_index(drop=True)


    va_data= va_data.drop(va_data[va_data['time'] > np.max(tr_data['time'])].index) #Discard the individuals from validation set whoose observed times are greater than the observed times of tranining data 
    te_data= te_data.drop(te_data[te_data['time'] > np.max(tr_data['time'])].index) #Discard the individuals from test set whoose observed times are greater than the observed times of tranining data 


    tr_protected_group1 = (tr_data[protected_group1]>65).astype(int) #Create a binary variable with two categories (more than 65 years and less or equal to 65 years)
    tr_protected_group1[tr_protected_group1==0]=2
    tr_data['protected_group1']=tr_protected_group1
    tr_protected_group2=tr_data[protected_group2] # Another protected attribute (e.g., race or gender)
    tr_data['protected_group2']=tr_protected_group2.values.argmax(1)+1

    va_protected_group1 = (va_data[protected_group1]>65).astype(int)
    va_protected_group1[va_protected_group1==0]=2
    va_data['protected_group1']=va_protected_group1
    va_protected_group2=va_data[protected_group2]
    va_data['protected_group2']=va_protected_group2.values.argmax(1)+1

    te_protected_group1 = (te_data[protected_group1]>65).astype(int)
    te_protected_group1[te_protected_group1==0]=2
    te_data['protected_group1']=te_protected_group1
    te_protected_group2=te_data[protected_group2]
    te_data['protected_group2']=te_protected_group2.values.argmax(1)+1

    #Standardization of the continuous variables
    features_idx = [standardized_list.index(feature) for feature in standardized_list] 
    train_data=tr_data.copy()
    val_data=va_data.copy()
    test_data=te_data.copy()
    scaler=StandardScaler().fit(tr_data.iloc[:, features_idx])
    train_data.iloc[:, features_idx]= scaler.transform(tr_data.iloc[:, features_idx])
    val_data.iloc[:, features_idx]= scaler.transform(va_data.iloc[:, features_idx])
    test_data.iloc[:, features_idx]= scaler.transform(te_data.iloc[:, features_idx])    

    
    ## Get the observed time and event status
    data_time_train = train_data['time']
    data_time_train = np.array(data_time_train,dtype='float32')
    data_time_val   = val_data['time']
    data_time_val   = np.array(data_time_val,dtype='float32')
    data_time_test  = test_data['time']
    data_time_test  = np.array(data_time_test,dtype='float32')
    data_event_train= train_data['status']
    data_event_train= np.array(data_event_train,dtype='float32')
    data_event_val  = val_data['status']
    data_event_val  = np.array(data_event_val,dtype='float32')
    data_event_test = test_data['status']
    data_event_test = np.array(data_event_test,dtype='float32')
    eval_time       = Evaltime(data_time_train) #Compute pre-specified time points
    
    ## Compute the pseudo values
    
    tr_pseudo = pseudo_values(train_data, eval_time)
    va_pseudo = pseudo_values(val_data, eval_time)
    te_pseudo = pseudo_values(test_data, eval_time)
    train_pseudo = np.array(tr_pseudo, dtype='float32')
    val_pseudo   = np.array(va_pseudo, dtype='float32')
    test_pseudo  = np.array(te_pseudo, dtype='float32') 
    
    ## Get the covariates
    data_X_train = train_data.drop(columns=['status','time', 'protected_group1', 'protected_group2']) #Covariates in train data
    data_X_val   = val_data.drop(columns=['status','time','protected_group1', 'protected_group2']) #Covariates in validation data
    data_X_test  = test_data.drop(columns=['status','time','protected_group1', 'protected_group2'])    #Covariates in test data
    
    data_X_train = np.array(data_X_train,dtype='float32')
    data_X_val   = np.array(data_X_val,dtype='float32')
    data_X_test  = np.array(data_X_test,dtype='float32')

    
    ## Prepare the covariates, observed time and event status for uncensored data
    train_data_uncen = train_data.loc[train_data['status']==1,:]
    test_data_uncen  = test_data.loc[test_data['status']==1,:]
    data_time_train_uncen = train_data_uncen['time']
    data_time_train_uncen = np.array(data_time_train_uncen,dtype='float32')
    data_time_test_uncen  = test_data_uncen['time']
    data_time_test_uncen  = np.array(data_time_test_uncen,dtype='float32')
    data_event_test_uncen = test_data_uncen['status']
    data_event_test_uncen = np.array(data_event_test_uncen,dtype='float32')
    data_X_test_uncen     = test_data_uncen.drop(columns=['status','time','protected_group1', 'protected_group2'])       
        
    ## Prepare the covariates, observed time and event status for censored data
    train_data_cen=train_data.loc[train_data['status']==0,:]
    test_data_cen=test_data.loc[test_data['status']==0,:]
    data_time_train_cen=train_data_cen['time']
    data_time_train_cen = np.array(data_time_train_cen,dtype='float32')
    data_time_test_cen=test_data_cen['time']
    data_time_test_cen = np.array(data_time_test_cen,dtype='float32')
    data_event_test_cen=test_data_cen['status']
    data_event_test_cen = np.array(data_event_test_cen,dtype='float32')
    data_X_test_cen=test_data_cen.drop(columns=['status','time','protected_group1', 'protected_group2'])      

    
    ## Create dictionaries of the covariates, observed time and event status for the protected groups of entire data, uncensored data and censored data
    protected_time_train={}
    protected_time_test={}
    protected_event_test={}
    protected_X_test={}        

    protected_time_train_uncen={}
    protected_time_test_uncen={}
    protected_event_test_uncen={}
    protected_X_test_uncen={}

    protected_time_train_cen={}
    protected_time_test_cen={}
    protected_event_test_cen={}
    protected_X_test_cen={}

    for i in protected_group2:  
        protected_time_train[i]=train_data['time'][train_data[i]==1]
        protected_time_train[i] = np.array(protected_time_train[i],dtype='float32')
        protected_time_test[i]=test_data['time'][test_data[i]==1]
        protected_time_test[i] = np.array(protected_time_test[i],dtype='float32')
        protected_event_test[i]=test_data['status'][test_data[i]==1]
        protected_event_test[i] = np.array(protected_event_test[i],dtype='float32')
        protected_X_test[i]=test_data.drop(columns=['status','time','protected_group1', 'protected_group2'])[test_data[i]==1] 
    
        
        protected_time_train_uncen[i]=train_data_uncen['time'][train_data_uncen[i]==1]
        protected_time_train_uncen[i] = np.array(protected_time_train_uncen[i],dtype='float32')
        protected_time_test_uncen[i]=test_data_uncen['time'][test_data_uncen[i]==1]
        protected_time_test_uncen[i] = np.array(protected_time_test_uncen[i],dtype='float32')
        protected_event_test_uncen[i]=test_data_uncen['status'][test_data_uncen[i]==1]
        protected_event_test_uncen[i] = np.array(protected_event_test_uncen[i],dtype='float32')
        protected_X_test_uncen[i]=test_data_uncen.drop(columns=['status','time','protected_group1', 'protected_group2'])[test_data_uncen[i]==1]   


        protected_time_train_cen[i]=train_data_cen['time'][train_data_cen[i]==1]
        protected_time_train_cen[i] = np.array(protected_time_train_cen[i],dtype='float32')
        protected_time_test_cen[i]=test_data_cen['time'][test_data_cen[i]==1]
        protected_time_test_cen[i] = np.array(protected_time_test_cen[i],dtype='float32')
        protected_event_test_cen[i]=test_data_cen['status'][test_data_cen[i]==1]
        protected_event_test_cen[i] = np.array(protected_event_test_cen[i],dtype='float32')
        protected_X_test_cen[i]=test_data_cen.drop(columns=['status','time','protected_group1', 'protected_group2'])[test_data_cen[i]==1]   
        
        
    return eval_time, test_data, data_X_train, data_X_val, data_X_test, data_X_test_uncen, data_X_test_cen, train_pseudo, val_pseudo, test_pseudo, data_time_train, data_time_train_uncen, data_time_train_cen, data_time_val, data_time_test, data_time_test_uncen,data_time_test_cen, data_event_train, data_event_val, data_event_test, data_event_test_uncen, data_event_test_cen, protected_X_test, protected_event_test, protected_time_test, protected_X_test_uncen, protected_X_test_cen, protected_time_train_uncen, protected_time_train_cen, protected_time_test_uncen,protected_time_test_cen, protected_event_test_uncen, protected_event_test_cen


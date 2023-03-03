import torch
from model import *
from fairness_measure import *
from utils import *


# ==============================================================================
#                             Train the FIDP model
# ==============================================================================    

def FIDP_train(dataloader, model, loss_fn, optimizer, ntimes, scale, lamda):
    """
    Train the FIDP model
    Arguments:
    dataloader -- Training dataloader
    model -- FIDP model 
    loss_fn--Objective function (e.g. Pseudo value based loss function)
    optimizer--Optimizer (e.g. Adam optimizer)
    ntimes--Number of pre-specified time points
    scale--scale parameter
    lambda--accuracy-fairness trade-off parameter
    Returns:
    Updated FIDP model 
    """ 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        pseudo_loss = loss_fn(pred, y)
        target_fairness = torch.tensor(0.0).to(device)
        IFloss=criterionHinge() ## Fairness penalty constraint
        R_loss = IFloss(target_fairness,pred,X, ntimes, scale)
        # calculate loss
        loss = pseudo_loss + lamda*R_loss        
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) 
    total_loss /= num_batches    
    return total_loss

# ==============================================================================
#                             Evaluate the FIDP model
# ==============================================================================

def FIDP_evaluate(dataloader, model, loss_fn, ntimes, scale, lamda):
    """
    Evaluate the FIDP model
    Arguments:
    dataloader -- Training dataloader
    model -- FIDP model 
    loss_fn--Objective function (e.g. Pseudo value based loss function)
    ntimes--Number of pre-specified time points
    scale--scale parameter
    lambda--accuracy-fairness trade-off parameter
    Returns:
    FIDP model evaluation
    """     
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_batches = len(dataloader)
    model.eval()
    test_loss=0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pseudo_loss = loss_fn(pred, y)
            target_fairness = torch.tensor(0.0).to(device)
            IFloss=criterionHinge() ## Fairness penalty constraint
            R_loss = IFloss(target_fairness,pred,X, ntimes, scale)
            # calculate loss
            loss = pseudo_loss + lamda*R_loss           
            test_loss += float(loss.item())

    test_loss /= num_batches

    return test_loss

# ==============================================================================
#                             Train the FIPNAM model
# ==============================================================================  

def FIPNAM_train(dataloader, model, loss_fn, optimizer, ntimes, scale, lamda):
    
    """
    Train the FIPNAM model
    Arguments:
    dataloader -- Training dataloader
    model -- FIPNAM model 
    loss_fn--Objective function (e.g. Pseudo value based loss function)
    optimizer--Optimizer (e.g. Adam optimizer)
    ntimes--Number of pre-specified time points
    scale--scale parameter
    lambda--accuracy-fairness trade-off parameter
    Returns:
    Updated FIPNAM model 
    """     
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred,_ = model(X)
        pseudo_loss = loss_fn(pred, y)
        target_fairness = torch.tensor(0.0).to(device)
        IFloss=criterionHinge() ## Fairness penalty constraint
        R_loss = IFloss(target_fairness,pred,X, ntimes, scale)
        # calculate loss
        loss = pseudo_loss + lamda*R_loss        
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) 
    total_loss /= num_batches    
    return total_loss
        
# ==============================================================================
#                             Evaluate the FIPNAM model
# ==============================================================================

def FIPNAM_evaluate(dataloader, model, loss_fn, ntimes, scale, lamda):
    """
    Evaluate the FIPNAM model
    Arguments:
    dataloader -- Training dataloader
    model -- FIPNAM model 
    loss_fn--Objective function (e.g. Pseudo value based loss function)
    ntimes--Number of pre-specified time points
    scale--scale parameter
    lambda--accuracy-fairness trade-off parameter
    Returns:
    FIPNAM model evaluation
    """       
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_batches = len(dataloader)
    model.eval()
    test_loss=0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred,_ = model(X)
            pseudo_loss = loss_fn(pred, y)
            target_fairness = torch.tensor(0.0).to(device)
            IFloss=criterionHinge() ## Fairness penalty constraint
            R_loss = IFloss(target_fairness,pred,X, ntimes, scale)
            # calculate loss
            loss = pseudo_loss + lamda*R_loss           
            test_loss += float(loss.item())

    test_loss /= num_batches

    return test_loss


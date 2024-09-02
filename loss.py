import torch
from torch import nn

def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

# Loss function for single-event and competing risks
def conditional_weibull_loss(model, x, t, E, elbo=True, copula=None):

    alpha = model.discount
    params = model.forward(x)

    t = t.reshape(-1,1).expand(-1, model.k)#(n, k)
    f_risks = []
    s_risks = []

    for i in range(model.risks):
        k = params[i][0]
        b = params[i][1]
        gate = nn.LogSoftmax(dim=1)(params[i][2])
        s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
        f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
        f = f + s
        s = (s + gate)
        s = torch.logsumexp(s, dim=1)#log_survival
        f = (f + gate)
        f = torch.logsumexp(f, dim=1)#log_density
        f_risks.append(f)#(n,3) each column for one risk
        s_risks.append(s)
    f = torch.stack(f_risks, dim=1)
    s = torch.stack(s_risks, dim=1)
    
    if model.risks == 4:
        if copula is None:
            p1 = f[:,0] + s[:,1] + s[:,2] + s[:,3]
            p2 = s[:,0] + f[:,1] + s[:,2] + s[:,3]
            p3 = s[:,0] + s[:,1] + f[:,2] + s[:,3]
            p4 = s[:,0] + s[:,1] + s[:,2] + f[:,3]
            e1 = (E == 0) * 1.0
            e2 = (E == 1) * 1.0
            e3 = (E == 2) * 1.0
            e4 = (E == 2) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3) + torch.sum(e4 * p4)
            loss = -loss/E.shape[0]
        else:
            raise NotImplementedError()
    elif model.risks == 3:
        if copula is None:
            p1 = f[:,0] + s[:,1] + s[:,2] 
            p2 = s[:,0] + f[:,1] + s[:,2]
            p3 = s[:,0] + s[:,1] + f[:,2]
            e1 = (E == 0) * 1.0
            e2 = (E == 1) * 1.0
            e3 = (E == 2) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
            loss = -loss/E.shape[0]
        else:
            S = torch.exp(s).clamp(0.001,0.999)
            p1 = f[:,0] + safe_log(copula.conditional_cdf("u", S))
            p2 = f[:,1] + safe_log(copula.conditional_cdf("v", S))
            p3 = f[:,2] + safe_log(copula.conditional_cdf("w", S))
            e1 = (E == 0) * 1.0
            e2 = (E == 1) * 1.0
            e3 = (E == 2) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) + torch.sum(e3 * p3)
            loss = -loss/E.shape[0]
    elif model.risks == 2:
        if copula is None:
            p1 = f[:,0] + s[:,1]
            p2 = s[:,0] + f[:,1]
            e1 = (E == 1) * 1.0
            e2 = (E == 0) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2) 
            loss = -loss/E.shape[0]
        else:
            S = torch.exp(s).clamp(0.001,0.999)
            p1 = f[:,0] + safe_log(copula.conditional_cdf("u", S))
            p2 = f[:,1] + safe_log(copula.conditional_cdf("v", S))
            e1 = (E == 1) * 1.0
            e2 = (E == 0) * 1.0
            loss = torch.sum(e1 * p1) + torch.sum(e2 * p2)
            loss = -loss/E.shape[0]
    elif model.risks == 1:#added single risk 
        e1 = (E == 1) * 1.0
        e2 = (E == 0) * 1.0
        loss = torch.sum(e1 * f[:,0]) + torch.sum(e2 * s[:,0]) 
        loss = -loss/E.shape[0]
    return loss
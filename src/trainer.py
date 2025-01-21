import math
import torch
from torch.utils.data import DataLoader, TensorDataset

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def loss_function(model1, model2, data, copula=None):
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2)
        p2 = LOG(f2) + LOG(s1)
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001, 0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    return -torch.mean(p1 * data['E'] + (1-data['E'])*p2)

def predict_survival_curve(model, x_test, time_bins, truth=False):
    device = torch.device("cpu")
    surv_estimate = torch.zeros((x_test.shape[0], time_bins.shape[0]), device=device)
    x_test = torch.tensor(x_test)
    time_bins = torch.tensor(time_bins)
    for i in range(time_bins.shape[0]):
        surv_estimate[:,i] = model.survival(time_bins[i], x_test)
    return surv_estimate, time_bins, time_bins.max()

def train_copula_model(model1, model2, train_data, val_data,
                       n_epochs, patience=1000, batch_size=32, lr=1e-3,
                       verbose=False, copula=None):
    # Enable gradients for models
    model1.enable_grad()
    model2.enable_grad()
    if copula is not None:
        copula.enable_grad()
    
    min_val_loss = 1000
    copula_grad_multiplier = 1.0
    copula_grad_clip = 1.0
    
    # Prepare optimizer
    optimizer_params = [{"params": model1.parameters(), "lr": lr},
                        {"params": model2.parameters(), "lr": lr}]
    if copula is not None:
        optimizer_params.append({"params": copula.parameters(), "lr": lr})
    
    optimizer = torch.optim.Adam(optimizer_params)

    # Create DataLoaders for mini-batching
    train_loader = DataLoader(TensorDataset(train_data['T'], train_data['X'], train_data['E']),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data['T'], val_data['X'], val_data['E']),
                            batch_size=batch_size, shuffle=False)

    stop_itr = 0
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Iterate over mini-batches
        for batch_idx, (T, X, E) in enumerate(train_loader):
            batch_data = {'T': T, 'X': X, 'E': E}
            loss = loss_function(model1, model2, batch_data, copula)
            loss.backward()

            # Handle copula gradients if copula is provided
            if copula is not None:
                for p in copula.parameters():
                    if p.grad is not None:
                        p.grad = (p.grad * copula_grad_multiplier).clip(
                            -1 * copula_grad_clip, 1 * copula_grad_clip
                        )
                optimizer.step()
                for p in copula.parameters():
                    if p < 0.01:
                        with torch.no_grad():
                            copula.theta.data.fill_(0.01)
            else:
                optimizer.step()

        # Validation phase
        with torch.no_grad():
            val_loss = 0.0
            for batch_idx, (T, X, E) in enumerate(val_loader):
                batch_data = {'T': T, 'X': X, 'E': E}
                val_loss += loss_function(model1, model2, batch_data, copula).item()

            val_loss /= len(val_loader)

            if verbose and epoch % 100 == 0:
                copula_theta = copula.theta if copula is not None else None
                print(f"Epoch {epoch}, Validation Loss: {val_loss} - Copula Theta: {copula_theta}")

            if not math.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr = 0
                best_c1 = model1.coeff.detach().clone()
                best_c2 = model2.coeff.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                best_theta = copula.theta.detach().clone() if copula is not None else None
                min_val_loss = val_loss
            else:
                stop_itr += 1
                if stop_itr == patience:
                    break

    # Restore best parameters
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.coeff = best_c1
    model2.coeff = best_c2
    if copula is not None:
        copula.theta = best_theta

    return model1, model2, copula, min_val_loss


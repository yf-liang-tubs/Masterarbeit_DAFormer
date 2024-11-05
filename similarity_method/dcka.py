import torch
import numpy as np

def remove_negative_eigenvalues(A):
    L, V = torch.linalg.eig(A)
    L[torch.view_as_real(L)[:,0]<0]=0
    return torch.view_as_real(V @ torch.diag_embed(L) @ torch.linalg.inv(V))[:,:,0]

def _get_ranks(x, device):
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp).to(device)
    
    ranks[tmp] = torch.arange(len(x)).to(device)
    return ranks

def spearman_correlation(x, y, device):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x, device)
    y_rank = _get_ranks(y, device)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

def pearsonr(x, y, device):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

class deltaRSA(object):
    def __init__(self, device):
        self.device = device
        
    def pairwise_distance(self, embedding):
        x_row = embedding.unsqueeze(-2)  # x_row = [n1, 1, x_dim]
        x_col = embedding.unsqueeze(-3)  # x_col = [1, n2, x_dim] 
        distance = torch.sum((x_row - x_col) ** 2, dim = -1)
        return distance
    
    def compute_deltarsa(self, X, Y, embedding_1, embedding_2):
        dist1 = self.pairwise_distance(self.normalize(X)).to(self.device)
        dist2 = self.pairwise_distance(self.normalize(Y)).to(self.device)
        
        input_dist_1 = self.pairwise_distance(self.normalize(embedding_1)).to(self.device).reshape(-1,1)
        
        input_dist_2 = self.pairwise_distance(self.normalize(embedding_2)).to(self.device).reshape(-1,1)



        res1, pve_1 = self.linear_residual(dist1, input_dist_1); res1 = res1.reshape(dist1.shape)
        res2, pve_2 = self.linear_residual(dist2, input_dist_2); res2 = res2.reshape(dist2.shape)


        rho = spearman_correlation(res1[torch.triu(torch.ones_like(res1), diagonal=1) == 1], res2[torch.triu(torch.ones_like(res2), diagonal=1) == 1], self.device)

        return rho

    def linear_residual(self, dist, input_dist):
        input_y = dist.reshape(-1,1)
        y_pred = input_dist.mm((input_dist.t().mm(input_dist)).inverse().mm(input_dist.t()).mm(input_y))
        res_sim = (input_y - y_pred).reshape(-1)
        pve = (1-res_sim.var()/input_y.var()).item()
#         print('Representation similarity explained by input similarity is: '+str(pve))
        return res_sim, pve
    
    def normalize(self, X):
        mean = X.mean(axis = 0).reshape(1,-1);
        X_mean = X - mean
        std = X_mean.norm()
        X_std = X_mean/std
        return X_std

class RSA(object):
    def __init__(self, device):
        self.device = device
        
    def pairwise_distance(self, embedding):
        x_row = embedding.unsqueeze(-2)  # x_row = [n1, 1, x_dim]
        x_col = embedding.unsqueeze(-3)  # x_col = [1, n2, x_dim] 
        distance = torch.sum((x_row - x_col) ** 2, dim = -1)
        return distance[torch.triu(torch.ones_like(distance), diagonal=1) == 1]
    
    def compute_rsa(self, X, Y):
        dist1 = self.pairwise_distance(self.normalize(X)).to(self.device)
        dist2 = self.pairwise_distance(self.normalize(Y)).to(self.device)
        return spearman_correlation(dist1, dist2, self.device)  
    
    def normalize(self, X):
        mean = X.mean(axis = 0).reshape(1,-1);
        X_mean = X - mean
        std = X_mean.norm()
        X_std = X_mean/std
        return X_std

class CKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        X = self.normalize(X); Y = self.normalize(Y)
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        X = self.normalize(X); Y = self.normalize(Y)
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
    
    def normalize(self, X):
        mean = X.mean(axis = 0).reshape(1,-1);
        X_mean = X - mean
        std = X_mean.norm()
        X_std = X_mean/std
        return X_std

class deltaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  


    def linear_HSIC(self, X, Y, embedding_1, embedding_2):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        
        input_dist_1 = torch.matmul(embedding_1, embedding_1.T).to(self.device).reshape(-1,1)
        
        input_dist_2 = torch.matmul(embedding_2, embedding_2.T).to(self.device).reshape(-1,1)
        
        res1, pve_1 = self.linear_residual(L_X, input_dist_1); res1 = res1.reshape(L_X.shape)
        res2, pve_2 = self.linear_residual(L_Y, input_dist_2); res2 = res2.reshape(L_Y.shape)
        

        hsic1 = torch.sum(self.centering(remove_negative_eigenvalues(res1)) * self.centering(remove_negative_eigenvalues(res2)))

        return hsic1

    def linear_CKA(self, X, Y, embedding_1, embedding_2):
        X = self.normalize(X); Y = self.normalize(Y)
        embedding_1 = self.normalize(embedding_1); embedding_2 = self.normalize(embedding_2);
        
        hsic1 = self.linear_HSIC(X, Y, embedding_1, embedding_2)
        var1 = self.linear_HSIC(X, X, embedding_1, embedding_1)
        var2 = self.linear_HSIC(Y, Y, embedding_2, embedding_2)

        return hsic1 / torch.sqrt(var1 * var2)

    
    def normalize(self, X):
        mean = X.mean(axis = 0).reshape(1,-1);
        X_mean = X - mean
        std = X_mean.norm()
        X_std = X_mean/std
        return X_std

    
    def linear_residual(self, dist, input_dist):
        input_y = dist.reshape(-1,1)
        y_pred = input_dist.mm((input_dist.t().mm(input_dist)).inverse().mm(input_dist.t()).mm(input_y))
        res_sim = (input_y - y_pred).reshape(-1)
        pve = (1-res_sim.var()/input_y.var()).item()
#         print('Representation similarity explained by input similarity is: '+str(pve))
        return res_sim, pve

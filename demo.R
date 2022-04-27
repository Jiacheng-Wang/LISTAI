#  #######################################################################
#       File-Name:      demo.R
#       Date:           Wed Apr 27 16:52:35 2022
#       Author:         JCW
#       Purpose:        
#       Input Files:    NONE
#       Output Files:   NONE
#       Data Output:    NONE
#       Dependencies:   NONE
#       Status:         In Progress
#  #######################################################################
set.seed(123)
n1 = n2 = n3 = 10
d1 = d2 = d3 = 2
p_sparse = 0.1
p = 0.1
sigma = 0.1

# create the auxiliary information matrices
A_true <- matrix(runif(n = d1*n1), ncol = d1)
B_true <- matrix(runif(n = d2*n2), ncol = d2)
C_true <- matrix(runif(n = d3*n3), ncol = d3)
# create the core tensor
G = rand_tensor(modes = c(d1,d2,d3))
sparse_index = array(runif(d1*d2*d3),dim = c(d1,d2,d3))
G@data[which(sparse_index<p_sparse,arr.ind = T)] = 0

# create the final tensor observation
lizt <- list(A_true, B_true, C_true)
tnsr_true = ttl(G,lizt,ms=c(1,2,3)) + as.tensor(array(rnorm(n1*n2*n3, mean = 0, sd = sigma),dim = c(n1,n2,n3)))
missing_index = array(runif(n1*n2*n3),dim = c(n1,n2,n3))
tnsr_obs = tnsr_true
tnsr_obs@data[which(missing_index< p, arr.ind = T)] = 0

omega = array(rep(1,n1*n2*n3),dim = c(n1,n2,n3))
omega[which(tnsr_obs@data == 0, arr.ind = T)] = 0

# perform the nested double ADMM algorithm to impute the missing values
res <- nested_double_admm(tnsr_obs,omega, A_true,B_true,C_true,lambda1 = 1, lambda2=0.01, beta1=0.8, rho = 2, tau=10, eta=5, max_iter = 100, tol = 1e-4)

# compute performance on observed entries and missing entries
sum(((res$E - tnsr_true@data)*omega)^2)/sum(omega) # 0.06055138
sum(((res$E - tnsr_true@data)*(1-omega))^2)/(prod(n1,n2,n3) - sum(omega)) # 0.07366219

#  #######################################################################
#       File-Name:      LISTAI_main.R
#       Date:           Wed Apr 27 08:59:41 2022
#       Author:         JCW
#       Purpose:        tensor completion with auxiliary information
#       Input Files:    NONE
#       Output Files:   NONE
#       Data Output:    NONE
#       Dependencies:   NONE
#       Status:         In Progress
#  #######################################################################
# load the necessary package
library(rTensor)
library(softImpute)
library(glmnet)

# perform the nested double ADMM algorithm
nested_double_admm <- function(tnsr,omega, A_true,B_true,C_true,
                               lambda1, lambda2, beta1, rho = 2, tau, eta, max_iter = 100, tol = 1e-4){
  n1 = tnsr@modes[1];n2 = tnsr@modes[2];n3 = tnsr@modes[3];
  A_true = cbind(A_true, rep(1, nrow(A_true)))
  B_true = cbind(B_true, rep(1, nrow(B_true)))
  C_true = cbind(C_true, rep(1, nrow(C_true)))
  d1 = ncol(A_true); d2 = ncol(B_true); d3 = ncol(C_true); 
  C = array(rnorm(n1*n2*n3), dim = c(n1,n2,n3))
  E = array(rnorm(n1*n2*n3), dim = c(n1,n2,n3))
  M1 = array(rnorm(n1*n2*n3), dim = c(n1,n2,n3))
  M2 = array(rnorm(n1*n2*n3), dim = c(n1,n2,n3))
  iter = 1
  fval_train = NULL
  fval_test = NULL
  fval_total = NULL
  while(iter < max_iter){
    G = G_update(n1,n2,n3,d1,d2,d3,A_true,B_true,C_true, lambda1, beta1, C, M1, E)
    C = C_update(n1,n2, n3,A_true, B_true, C_true, G,E,M1,beta1) 
    E = E_update_total(tnsr, omega,n1,n2,n3,A_true, B_true, C_true, E, G, C, M1, M2, beta1, tau, eta,lambda2)@data
    M1 = M1_update(A_true, B_true, C_true, M1,beta1, C,G,E)
    M2 = M2_update(tnsr,A_true, B_true, C_true, M2,beta1,E)
    beta1 = min(10, rho*beta1)
    fval_train = append(fval_train, rTensor::fnorm(as.tensor((E - tnsr)@data*omega))/sum(omega))
    if (iter > 1){
      if(abs(fval_train[iter] - fval_train[iter-1])/fval_train[iter-1] < tol){break}
    }
    print(paste('iter:', iter, 'err', fval_train[iter]))
    iter = iter+1
  }
  return(list(G = G,
              E = E,
              C = C,
              err = fval_train))
}

# dependent functions
# update the auxiliary variable tensor
# C = [G; X,Y,Z] - E
C_update <- function(n1,n2, n3,A_true, B_true, C_true, G, E, M1, beta1){
  C_auxilary = array(rep(0, n1*n2*n3),dim = c(n1,n2,n3))
  C_auxilary = beta1/(beta1+1)*(ttl(G,list(A_true, B_true, C_true),ms=c(1,2,3))@data - E - M1/beta1)
  return(C_auxilary)
}
# update the core sparse tensor
G_update <- function(n1,n2,n3,d1,d2,d3,A_true,B_true,C_true,lambda1, beta1, C, M1, E){
  fit<- glmnet(kronecker_list(list(C_true,B_true,A_true)), vec(as.tensor(C + M1/beta1 + E)),family = 'gaussian')
  g_vec <- coef(fit, s= lambda1/(beta1*n1*n2*n3))
  G = as.tensor(array(g_vec,dim = c(d1,d2,d3)))
  return(G)
}
# update the low rank tensor E
E_update <- function(tnsr_true, omega, n1,n2,n3, A_true, B_true, C_true, E, G, C, M1, M2, z1,z2,z3, tau, beta1,eta, alpha1, alpha2, alpha3){
  f2 = E - (ttl(G,list(A_true, B_true, C_true),ms=c(1,2,3))@data - C-M1/beta1)
  f3 = (E - tnsr_true + M2/beta1)*omega
  response = E - (f2+f3)/(2*tau)
  E = (2*beta1*tau*response + eta*(fold(z1-alpha1, row_idx = 1, col_idx = c(2,3), modes = c(n1,n2,n3))+
                                     fold(z2-alpha2, row_idx = 2, col_idx = c(1,3), modes = c(n1,n2,n3))+
                                     fold(z3-alpha3, row_idx = 3, col_idx = c(1,2), modes = c(n1,n2,n3))))/(2*beta1*tau + 3*eta)
  return(list(E = E,
              r = response))
}
z1_update <- function(E, alpha1,eta, lambda2){
  fit = svd.als(unfold(E, row_idx = 1, col_idx = c(2,3))@data - alpha1, lambda = 2*lambda2/(3*eta), maxit = 5000)
  if(length(fit$d) == 1){
    return(fit$u%*%t(fit$v))
  }else{
    return(fit$u%*%diag(fit$d)%*%t(fit$v))
  }
}

z2_update <- function(E, alpha2,eta, lambda2){
  fit = svd.als(unfold(E, row_idx = 2, col_idx = c(1,3))@data - alpha2, lambda = 2*lambda2/(3*eta), maxit = 5000)
  if(length(fit$d) == 1){
    return(fit$u%*%t(fit$v))
  }else{
    return(fit$u%*%diag(fit$d)%*%t(fit$v))
  }
}

z3_update <- function(E, alpha3,eta,lambda2){
  fit = svd.als(unfold(E, row_idx = 3, col_idx = c(1,2))@data - alpha3, lambda = 2*lambda2/(3*eta), maxit = 5000)
  if(length(fit$d) == 1){
    return(fit$u%*%t(fit$v))
  }else{
    return(fit$u%*%diag(fit$d)%*%t(fit$v))
  }
}

E_update_total <- function(tnsr_true, omega, n1,n2,n3, A_true, B_true, C_true, E, G, C, M1, M2,beta1, tau, eta,lambda2, max_iter = 500, tol = 1e-3){
  # E return as a tensor
  z1 = matrix(rnorm(n1*n2*n3), nrow = n1)
  z2 = matrix(rnorm(n1*n2*n3), nrow = n2)
  z3 = matrix(rnorm(n1*n2*n3), nrow = n3)
  alpha1 = matrix(rnorm(n1*n2*n3), nrow = n1)
  alpha2 = matrix(rnorm(n1*n2*n3), nrow = n2)
  alpha3 = matrix(rnorm(n1*n2*n3), nrow = n3)
  iter = 1
  while(iter < max_iter){
    ans = E_update(tnsr_true, omega, n1,n2,n3,A_true, B_true, C_true,E, G, C, M1, M2, z1,z2,z3, tau, beta1,eta, alpha1, alpha2, alpha3)
    E = ans$E
    obj_previous = lambda2/3*(nuclear(z1) + nuclear(z2)+nuclear(z3)) + beta1*tau*rTensor::fnorm(E -ans$r)
    z1 = z1_update(E, alpha1,eta,lambda2)
    z2 = z2_update(E, alpha2,eta,lambda2)
    z3 = z3_update(E, alpha3,eta,lambda2)
    alpha1 = alpha1 + unfold(E, row_idx = 1, col_idx = c(2,3))@data - z1
    alpha2 = alpha2 + unfold(E, row_idx = 2, col_idx = c(1,3))@data - z2
    alpha3 = alpha3 + unfold(E, row_idx = 3, col_idx = c(1,2))@data - z3
    eta = eta/sd(ans$r@data)
    obj_new = lambda2/3*(nuclear(z1) + nuclear(z2)+nuclear(z3)) + beta1*tau*rTensor::fnorm(E -ans$r)
    if (max(obj_previous - obj_new)<tol){break}
    obj_previous = obj_new
    iter = iter + 1
  }
  return(E)
}
# update lagrangian multiplier M1
M1_update <- function(A_true, B_true, C_true, M1_old, beta1, C, G, E){
  M1 <- M1_old + beta1*(C - ttl(G,list(A_true, B_true, C_true),ms=c(1,2,3))@data + E)
  return(M1)
}

M2_update <- function(tnsr_obs, A_true, B_true, C_true, M2_old, beta1, E){
  M1 <- M2_old + beta1 * (omega*E - tnsr_obs)
  return(M1)
}

# Step 5: main function for nested double admm algorithm


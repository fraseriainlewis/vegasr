
##############################################################################################
##############################################################################################
### Prepare model inputs
fn_create_data_1<-function(seed){
  set.seed(seed)
  # Set up data
  rr_k_ctrl <- c(0.20)        # control response rate for each basket
  rr_k_trt <- c(0.40)         # treatment response rate for each basket

  K<-length(rr_k_ctrl)        # number of baskets

  N_k_ctrl <- rep(100, K)     # number of control participants per basket
  N_k_trt <- rep(100, K)      # number of treatment participants per basket
  N_k <- N_k_ctrl + N_k_trt   # number of participants per basket (both arms combined)
  N <- sum(N_k)               # total sample size
  k_vec <- rep(1:K, N_k)      # N x 1 vector of basket indicators (1 to K)

  z_vec<-NULL;
  y<-NULL;
  for(i in 1:K){ # for each basket repeat 0-control 1-trt according to the specifc Ns
    z_vec<-c(z_vec,rep(0:1,c(N_k_ctrl[i],N_k_trt[i]))) # treatment/control indicator
    y<-c(y,
         c(stats::rbinom(N_k_ctrl[i],1,rr_k_ctrl[i]), # bernoulli for control
           stats::rbinom(N_k_trt[i],1,rr_k_trt[i]))) #           for trt
  }

  thedata<-data.frame(y,basketID=k_vec,Treatment=z_vec)
  # response
  y<-matrix(data=as.numeric(thedata$y),ncol=1)
  # treatment
  treat<-matrix(data=as.numeric(thedata$Treatment),ncol=1)
  # basket
  #basket<-matrix(data=as.numeric(thedata$Treatment),ncol=1)
  return(list(y=y,treat=treat))

}
##############################################################################################
##############################################################################################
## Define log posterior inclding change of variables
fn_log_post_1<-function(theta,      # matrix Batch x M
                        y,          # matrix N x 1
                        treat,      # matrix N x 1
                        shiftby,    # scalar - no scaling
                        uselog           # scalar - default return exp(log())
){

  # this is a trimming function to avoid extreme end of integral limits,
  # same as -inf,+inf
  theta0=pmin(pmax(theta[,1], -0.9999), 0.9999)
  theta1=pmin(pmax(theta[,2], -0.9999), 0.9999)
  theta2=pmin(pmax(theta[,3], -0.9999), 0.9999)
  theta3=pmin(pmax(theta[,4], -0.9999), 0.9999)
  theta4=pmin(pmax(theta[,5], 0.0001), 0.9999)
  theta5=pmin(pmax(theta[,6], 0.0001), 0.9999)

  jacobianL = (
    log1p(theta0^2) - 2.0*log1p(-(theta0^2))
    + log1p(theta1^2) - 2.0*log1p(-(theta1^2))
    + log1p(theta2^2) - 2.0*log1p(-(theta2^2))
    + log1p(theta3^2) - 2.0*log1p(-(theta3^2))
    + log1p(theta4^2) - 2.0*log1p(-(theta4^2))
    + log1p(theta5^2) - 2.0*log1p(-(theta5^2))

  )

  a0=theta0/(1-theta0^2)
  a1=theta1/(1-theta1^2)
  mu0=theta2/(1-theta2^2)
  mu1=theta3/(1-theta3^2)
  sigma0=theta4/(1-theta4^2)
  sigma1=theta5/(1-theta5^2)

  # eta=a0+a1*treat # (10,3) where a0 and a1 = (3,) and treat is (10,1)
  #                                                broadcasts to (10,3)
  # R doesn't have auto broadcast but this is equivalent
  eta <- sweep(treat %*% rbind(a1), 2, a0, "+")
  # logL = sum_i^n ( y_i*(a0+a1*T_i) - log(1+ exp(a0+a1*T_i) )  )
  logL = apply(sweep(eta,1,y,"*") - log1p(exp(eta)),2,sum)

  # now add the priors
  prior_a0 = stats::dnorm(a0,mean=mu0,sd=sigma0,log=TRUE)
  prior_a1 = stats::dnorm(a1,mean=mu1,sd=sigma1,log=TRUE)
  prior_mu0 = stats::dnorm(mu0,mean=0.,sd=2.5,log=TRUE)
  prior_mu1 = stats::dnorm(mu1,mean=0.,sd=2.5,log=TRUE)
  prior_sigma0 = extraDistr::dhnorm(sigma0, sigma=2.5,log=TRUE)
  prior_sigma1 = extraDistr::dhnorm(sigma1, sigma=2.5,log=TRUE)

  logDens = logL + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 +
    prior_sigma0 + prior_sigma1
  logPost = logDens + jacobianL

  if(uselog==1.){ # search phase for max - keep in log
    return(logPost - shiftby[1])
  } else return(exp(logPost - shiftby[1]) ) # integrand eval - use raw

}


if(FALSE){ # to test
  vegasr:::fn_create_data_1(99999)
  # response
  y<-matrix(data=as.numeric(thedata$y),ncol=1)
  # treatment
  treat<-matrix(data=as.numeric(thedata$Treatment),ncol=1)
  # matrix of samole parameter values - nrow is batch, ncol is model dimension
  dummytheta<-matrix(data=c(-0.11,-0.13,0.15,0.11,0.051,0.052,
                            -0.12,-0.1,0.17,0.12,0.052,0.051,
                            -0.13,-0.11,0.11,0.19,0.053,0.054
  ),ncol=6,byrow=TRUE)

  ## test if log_posterior works - pass matrix of parameter values
  vegasr:::fn_log_post_1(theta=dummytheta,y=y,treat=treat,shiftby=0,uselog=1.)

}

##############################################################################################
##############################################################################################
## Define log posterior inclding change of variables
fn_marg_1_1<-function(theta,      # matrix Batch x M
                      y,          # matrix N x 1
                      treat,      # matrix N x 1
                      shiftby,    # scalar - no scaling
                      uselog,     # scalar - default return exp(log())
                      z           # z of f(z) in marginal
){

  # this function is almost same as for log evidence but one less dimension so need to
  # comment out extra dim and adjust indexes in arrays
  # this is a trimming function to avoid extreme end of integral limits,
  # same as -inf,+inf
  #theta0=pmin(pmax(theta[,1], -0.9999), 0.9999)
  theta1=pmin(pmax(theta[,2-1], -0.9999), 0.9999)
  theta2=pmin(pmax(theta[,3-1], -0.9999), 0.9999)
  theta3=pmin(pmax(theta[,4-1], -0.9999), 0.9999)
  theta4=pmin(pmax(theta[,5-1], 0.0001), 0.9999)
  theta5=pmin(pmax(theta[,6-1], 0.0001), 0.9999)

  jacobianL = (
    #log1p(theta0^2) - 2.0*log1p(-(theta0^2))
    #+
    log1p(theta1^2) - 2.0*log1p(-(theta1^2))
    + log1p(theta2^2) - 2.0*log1p(-(theta2^2))
    + log1p(theta3^2) - 2.0*log1p(-(theta3^2))
    + log1p(theta4^2) - 2.0*log1p(-(theta4^2))
    + log1p(theta5^2) - 2.0*log1p(-(theta5^2))

  )

  #a0=theta0/(1-theta0^2)
  a0=rep(z,length(theta1)) # z is passed
  a1=theta1/(1-theta1^2)
  mu0=theta2/(1-theta2^2)
  mu1=theta3/(1-theta3^2)
  sigma0=theta4/(1-theta4^2)
  sigma1=theta5/(1-theta5^2)

  # eta=a0+a1*treat # (10,3) where a0 and a1 = (3,) and treat is (10,1)
  #                                                broadcasts to (10,3)
  # R doesn't have auto broadcast but this is equivalent
  eta <- sweep(treat %*% rbind(a1), 2, a0, "+")
  # logL = sum_i^n ( y_i*(a0+a1*T_i) - log(1+ exp(a0+a1*T_i) )  )
  logL = apply(sweep(eta,1,y,"*") - log1p(exp(eta)),2,sum)

  # now add the priors
  prior_a0 = stats::dnorm(a0,mean=mu0,sd=sigma0,log=TRUE)
  prior_a1 = stats::dnorm(a1,mean=mu1,sd=sigma1,log=TRUE)
  prior_mu0 = stats::dnorm(mu0,mean=0.,sd=2.5,log=TRUE)
  prior_mu1 = stats::dnorm(mu1,mean=0.,sd=2.5,log=TRUE)
  prior_sigma0 = extraDistr::dhnorm(sigma0, sigma=2.5,log=TRUE)
  prior_sigma1 = extraDistr::dhnorm(sigma1, sigma=2.5,log=TRUE)

  logDens = logL + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 +
    prior_sigma0 + prior_sigma1
  logPost = logDens + jacobianL

  if(uselog==1.){ # search phase for max - keep in log
    return(logPost - shiftby[1])
  } else return(exp(logPost - shiftby[1]) ) # integrand eval - use raw

}

if(FALSE){ # to test
  vegasr:::fn_create_data_1(99999)
  # response
  y<-matrix(data=as.numeric(thedata$y),ncol=1)
  # treatment
  treat<-matrix(data=as.numeric(thedata$Treatment),ncol=1)
  # matrix of samole parameter values - nrow is batch, ncol is model dimension
  dummytheta<-matrix(data=c(-0.13,0.15,0.11,0.051,0.052,
                            -0.1,0.17,0.12,0.052,0.051,
                            -0.11,0.11,0.19,0.053,0.054
  ),ncol=5,byrow=TRUE)

  ## test if log_posterior works - pass matrix of parameter values
  vegasr:::fn_marg_1_1(theta=dummytheta,y=y,treat=treat,shiftby=0,uselog=1.)

}



##############################################################################################
##############################################################################################




vegasBayesEvidence <- function(f, lower,upper, nitn_warm = 10, neval_warm = 1000,
                  nitn = 10, neval = 1000, errTol=1,maxIter=5,
                  seed = 99999,nsearch=1000,
                  extra_args = list()) {


  set.seed(seed)
  start<-lower
  stop<-upper
  searchPts <- mapply(seq, from = start, to = stop,
                      MoreArgs = list(length.out = nsearch))
  for(i in 1:length(start)){searchPts[,i]<-searchPts[sample(1:nrow(searchPts)),i]}

  #print(extra_args)
  #print(searchPts)
  extra_args$uselog[1]<-1.0 # force this
  extra_args$shiftby[1]<-0.0 # force this
  mymax<-max(do.call(f,c(list(searchPts),extra_args)))

  # location shift - approx max over the integration domain
  cat("max log value found = ",mymax,"\n")

  extra_args$uselog[1]<-0.0 #force this
  extra_args$shiftby[1]<-mymax

  result<-vegas(f=f, lower=lower,upper=upper, nitn_warm = nitn_warm, neval_warm = neval_warm,
        nitn = nitn, neval = neval, errTol=errTol,maxIter=maxIter,
        seed = seed,
        extra_args = extra_args)

  log_evidence = mymax + log(result$mean)
  cat("log evidence = ",log_evidence,"\n")
  # should be around -129.4
  return(log_evidence)


}



#lower=c(-5,-5,-5); upper=c(5,5,5);nitn_warm = 10;neval_warm = 10000;nitn = 10; neval = 10000;
#errTol=0.1;maxIter=20;seed=99999;nsearch=10;extra_args=list(mu=mu,cov=cov);



### Prepare model inputs
set.seed(99999)
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
       c(rbinom(N_k_ctrl[i],1,rr_k_ctrl[i]), # bernoulli for control
         rbinom(N_k_trt[i],1,rr_k_trt[i]))) #           for trt
}

thedata<-data.frame(y,basketID=k_vec,Treatment=z_vec)


# response
y<-matrix(data=as.numeric(thedata$y),ncol=1)
# treatment
treat<-matrix(data=as.numeric(thedata$Treatment),ncol=1)
# matrix of samole parameter values - nrow is batch, ncol is model dimension
dummytheta<-matrix(data=c(-0.11,-0.13,0.15,0.11,0.051,0.052,
                          -0.12,-0.1,0.17,0.12,0.052,0.051,
                          -0.13,-0.11,0.11,0.19,0.053,0.054
),ncol=6,byrow=TRUE)

## Define log posterior inclding change of variables
r_compute_log_lik<-function(theta,      # matrix Batch x M
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
  prior_a0 = dnorm(a0,mean=mu0,sd=sigma0,log=TRUE)
  prior_a1 = dnorm(a1,mean=mu1,sd=sigma1,log=TRUE)
  prior_mu0 = dnorm(mu0,mean=0.,sd=2.5,log=TRUE)
  prior_mu1 = dnorm(mu1,mean=0.,sd=2.5,log=TRUE)
  prior_sigma0 = dhnorm(sigma0, sigma=2.5,log=TRUE)
  prior_sigma1 = dhnorm(sigma1, sigma=2.5,log=TRUE)

  logDens = logL + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 +
    prior_sigma0 + prior_sigma1
  logPost = logDens + jacobianL

  if(uselog==1.){ # search phase for max - keep in log
    return(logPost - shiftby[1])
  } else return(exp(logPost - shiftby[1]) ) # integrand eval - use raw

}

## test if log_posterior works - pass matrix of parameter values
r_compute_log_lik(theta=dummytheta,y=y,treat=treat,shiftby=0,uselog=1.)




result_logEv<-vegasBayesEvidence(f=r_compute_log_lik,
              lower=c(-1,-1,-1,-1,0.0001,0.0001),
              upper=c(1,1,1,1,1,1),
              nitn_warm = 10, neval_warm = 10000,
              nitn = 10, neval = 10000,
              errTol=1,maxIter=10,seed=99999,nsearch=10000,
              extra_args=list(
                y=y,treat=treat,shiftby=0,uselog=1.))
print(result_logEv)

##############################################################################################
##############################################################################################
vegasBayesPosterior <- function(f, lower,upper, nitn_warm = 10, neval_warm = 1000,
                                nitn = 10, neval = 1000, errTol=1,maxIter=5,seed=99999,
                                nsearch=1000,
                                log_evidence,
                                extra_args=list() # x must have z in f(z)
){

  set.seed(seed)
  start<-lower
  stop<-upper
  searchPts <- mapply(seq, from = start, to = stop,
                      MoreArgs = list(length.out = nsearch))
  for(i in 1:length(start)){searchPts[,i]<-searchPts[sample(1:nrow(searchPts)),i]}

  #print(extra_args)
  #print(searchPts)
  if(!"uselog"%in%names(extra_args)){stop("extra_args list must have named member 'uselog'. see help page")}
  if(!"shiftby"%in%names(extra_args)){stop("extra_args list must have named member 'shiftby'. see help page")}
  if(!"z"%in%names(extra_args)){stop("extra_args list must have named member 'uselog'. see help page")}

  extra_args$uselog[1]<-1.0 # force this - use log return values
  extra_args$shiftby[1]<-0.0 # force this - no shifting, might get some -inf which is fine

  ## join search points into args as first argument and evaluation all search point
  mymax<-max(do.call(f,c(list(searchPts),extra_args)))

  # location shift - approx max over the integration domain
  cat("max log value found = ",mymax,"\n")

  extra_args$uselog[1]<-0.0 #force this don't use log
  extra_args$shiftby[1]<-mymax

  result<-vegas(f=f, lower=lower,upper=upper, nitn_warm = nitn_warm, neval_warm = neval_warm,
                nitn = nitn, neval = neval, errTol=errTol,maxIter=maxIter,
                seed = seed,
                extra_args = extra_args # this has z=value
  )

  unstd_log_margZ = mymax + log(result$mean)

  prob_dens=exp(unstd_log_margZ-log_evidence)
#cat("f(z) = ",prob_dens,"\n")

return(prob_dens)

}
#####
#####
### Prepare model inputs
set.seed(99999)
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
       c(rbinom(N_k_ctrl[i],1,rr_k_ctrl[i]), # bernoulli for control
         rbinom(N_k_trt[i],1,rr_k_trt[i]))) #           for trt
}

thedata<-data.frame(y,basketID=k_vec,Treatment=z_vec)


# response
y<-matrix(data=as.numeric(thedata$y),ncol=1)
# treatment
treat<-matrix(data=as.numeric(thedata$Treatment),ncol=1)
# matrix of samole parameter values - nrow is batch, ncol is model dimension
dummytheta<-matrix(data=c(-0.11,-0.13,0.15,0.11,0.051,0.052,
                          -0.12,-0.1,0.17,0.12,0.052,0.051,
                          -0.13,-0.11,0.11,0.19,0.053,0.054
),ncol=6,byrow=TRUE)

## Define log posterior inclding change of variables
r_compute_log_lik_marg<-function(theta,      # matrix Batch x M
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
  prior_a0 = dnorm(a0,mean=mu0,sd=sigma0,log=TRUE)
  prior_a1 = dnorm(a1,mean=mu1,sd=sigma1,log=TRUE)
  prior_mu0 = dnorm(mu0,mean=0.,sd=2.5,log=TRUE)
  prior_mu1 = dnorm(mu1,mean=0.,sd=2.5,log=TRUE)
  prior_sigma0 = dhnorm(sigma0, sigma=2.5,log=TRUE)
  prior_sigma1 = dhnorm(sigma1, sigma=2.5,log=TRUE)

  logDens = logL + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 +
    prior_sigma0 + prior_sigma1
  logPost = logDens + jacobianL

  if(uselog==1.){ # search phase for max - keep in log
    return(logPost - shiftby[1])
  } else return(exp(logPost - shiftby[1]) ) # integrand eval - use raw

}

## test if log_posterior works - pass matrix of parameter values
r_compute_log_lik_marg(theta=dummytheta,y=y,treat=treat,shiftby=0,uselog=1.,z=-1.)




mymarg<-vegasBayesPosterior(f=r_compute_log_lik_marg,
                           lower=c(-1,-1,-1,0.0001,0.0001),
                           upper=c(1,1,1,1,1),
                           nitn_warm = 10, neval_warm = 10000,
                           nitn = 10, neval = 10000,
                           errTol=1,maxIter=10,seed=99999,nsearch=10000,
                           log_evidence = result_logEv,
                           extra_args=list(
                             y=y,treat=treat,shiftby=0,uselog=1.,z=-2.))
print(mymarg)


library(foreach)
library(doParallel)
library(extraDistr)
library(tictoc)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

myz<-seq(-2.5,-0.,len=50)
tic("Parallel Vegas Loop") # Start timer with a label
f_z<-foreach(z= myz,.packages = c("extraDistr", "vegasr")) %dopar% {
 vegasBayesPosterior(f=r_compute_log_lik_marg,
                      lower=c(-1,-1,-1,0.0001,0.0001),
                      upper=c(1,1,1,1,1),
                      nitn_warm = 10, neval_warm = 10000,
                      nitn = 10, neval = 10000,
                      errTol=1,maxIter=10,seed=99999,nsearch=10000,
                      log_evidence = result_logEv,
                      extra_args=list(
                        y=y,treat=treat,shiftby=0,uselog=1.,z=z))

 }

stopCluster(cl)
toc() # Stops timer and prints: "Parallel Vegas Loop: 45.23 sec elapsed"

plot(myz,f_z,main="Vegas R",col="blue",lwd=2,type="n")
points(myz,f_z,col="brown",pch=21,bg="brown")
f_interp <- splinefun(myz, f_z, method = "fmm")
lines(myz,f_interp(myz),col="magenta")

f_interp <- smooth.spline(myz, f_z)
lines(f_interp,col="green")

#plot(myz,f_z,type="b")


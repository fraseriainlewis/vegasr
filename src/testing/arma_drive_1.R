library(RcppArmadillo)
library(RcppEigen)
library(RcppParallel)
library(tictoc)
## provides dmvnorm_aram(x,my,cov)

Rcpp::sourceCpp("src/testing/eigen_v1.cpp")
Rcpp::sourceCpp("src/testing/test.cpp")

thedata<-vegasr:::fn_create_data_5(99999)
theta   <- matrix(data=rep(0.1,length=4*14), ncol = 14) # 5 intercept + 5 slope + 4 hyper
theta<-jitter(theta)
#eigen_norm2_logpdf(theta, rep(as.double(1.0),4),rep(as.double(1.0),4))

library(vegasr)
# now setup python environment
vegas_initialize()
vegasr::eigen_fn_log_post_5(theta, thedata$y, thedata$treat, thedata$basket,0.0, 1.0)

eigen_fn_log_post_5_par(theta, thedata$y, thedata$treat, thedata$basket,0.0, 1.0)


fn_log_post_K(theta, thedata$y, thedata$treat, thedata$basket,0.0, 1.0,K=5)


set.seed(99999)
K <- length(unique(thedata$basket))
lower <- c(rep(-0.9, 2*K), -0.9, -0.9, 1e-2, 1e-2)
upper <- c(rep( 0.9, 2*K),  0.9,  0.9,   0.9,   0.9)
start<-lower
stop<-upper
searchPts <- mapply(seq, from = start, to = stop,
                    MoreArgs = list(length.out = 10000))
for(i in 1:length(start)){searchPts[,i]<-searchPts[sample(1:nrow(searchPts)),i]}
extra_args=list(
  y=thedata$y,
  treat=thedata$treat,
  basket = thedata$basket,
  shiftby=0,uselog=1.)

eigen_fn_log_post_5(searchPts[1:2,], thedata$y, thedata$treat, thedata$basket,0.0, 1.0)

eigen_fn_log_post_5_par(searchPts[1:2,], thedata$y, thedata$treat, thedata$basket,0.0, 1.0)

fn_log_post_K(searchPts[1:2,], thedata$y, thedata$treat, thedata$basket,0.0, 1.0,K=5)


res<-(do.call(eigen_fn_log_post_5,c(list(searchPts),extra_args)));print(max(res))
res<-(do.call(eigen_fn_log_post_5_par,c(list(searchPts),extra_args)));print(max(res))

res<-(do.call(fn_log_post_K,c(list(searchPts),extra_args)));print(max(res))


library(vegasr)
# now setup python environment
vegas_initialize() # this needed called once per session after library(vegas)
library(tictoc)


K <- length(unique(thedata$basket))

lower <- c(rep(-0.9, 2*K), -0.9, -0.9, 1e-2, 1e-2)
upper <- c(rep( 0.9, 2*K),  0.9,  0.9,   0.9,   0.9)

tic()
result_logEv <- vegasBayesEvidence(
  f = vegasr::eigen_fn_log_post_5,
  lower = lower, upper = upper,
  nitn_warm = 10, neval_warm = 10000,
  nitn = 10, neval = 10000,
  errTol = 0.1, maxIter = 10, seed = 99999, nsearch = 10000,
  extra_args=list(
    y=thedata$y,
    treat=thedata$treat,
    basket = thedata$basket,
    shiftby=0,uselog=1.)
)
cat("log evidence = ",result_logEv,"\n")
toc()

tic()
result_logEv <- vegasBayesEvidence(
  f = eigen_fn_log_post_5_par,
  lower = lower, upper = upper,
  nitn_warm = 10, neval_warm = 10000,
  nitn = 10, neval = 10000,
  errTol = 0.1, maxIter = 10, seed = 99999, nsearch = 10000,
  extra_args=list(
    y=thedata$y,
    treat=thedata$treat,
    basket = thedata$basket,
    shiftby=0,uselog=1.)
)
cat("log evidence = ",result_logEv,"\n")
toc()

tic()
result_logEv <- vegasBayesEvidence(
  f = fn_log_post_K,
  lower = lower, upper = upper,
  nitn_warm = 10, neval_warm = 10000,
  nitn = 10, neval = 10000,
  errTol = 0.1, maxIter = 10, seed = 99999, nsearch = 10000,
  extra_args=list(
    y=thedata$y,
    treat=thedata$treat,
    basketID = thedata$basket,
    shiftby=0,uselog=1.)
)
cat("log evidence = ",result_logEv,"\n")
toc()







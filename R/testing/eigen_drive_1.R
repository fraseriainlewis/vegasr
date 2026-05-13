library(RcppArmadillo)
## provides dmvnorm_aram(x,my,cov)
Rcpp::sourceCpp("src/testing/eigen_v1.cpp")

theta   <- matrix(data=rep(0.1,length=4*6), ncol = 6)

arma_fn_log_post_1(theta, thedata$y, thedata$treat,0.0, 1.0)


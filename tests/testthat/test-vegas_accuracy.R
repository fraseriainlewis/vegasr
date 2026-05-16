##################################################################################
# This file contains numerical tests to check that the wrapper is functioning correctly
# i.e. an R function which defines an integrand is passed to python vegas and
# vegas computes the correct numerical value (within tolerance) when compared to
# existing available R fuctions that numerical estimate same integrands
# the mvtnorm library is used for density and distribution functions
##################################################################################
test_that("vegas is wrapped correctly - nunmerical check 1", {
  # check Vegas is working correctly by using it to integrate existing density
  # functions i.e. a dXXX() function in R, and check against the pXXX()
  # function. use arbitrary upper and lower bounds
  # Specific example1 a 3-D MVM density
  skip_if_not_installed("mvtnorm")
  skip_on_cran()  # This test will be skipped on CRAN servers
  skip_on_ci()

  # Use library(mvtnorm) first for 3-D Multivariate Normal Density
  library(mvtnorm)
  mu<-matrix(c(0.5, -0.2, 0.1),nrow=1) # note matrix
  cov<-matrix(data=c(
    1.0, 0.5, 0.2,
    0.5, 1.2, 0.3,
    0.2, 0.3, 0.8),ncol=3,byrow=FALSE)

  # use built-in pmvnorm() - distribution function
  lower=c(-0.5,-0.5,-0.5); upper=c(2.,1.,3.)
  r_result<-pmvnorm(lower=lower,upper=upper,
                       mean=as.numeric(mu), # coercion as needs vector
                       sigma=cov)

  ## now use vegas to compute same integral
  library(vegasr)
  vegas_initialize()

  # call the built-in density function and use vegas to integrate it
  myf<-function(x,mu,cov){
    res<-dmvnorm(x,
                 mean = mu, # this is a 1-row matrix, dmvnorm accepts this
                 sigma=cov)
    return(res)
  }

  ## See help page for descriptions of warm and nitn and neval.
  vegas_result<-vegas(f=myf,
                      lower=lower, upper=upper,
                      nitn_warm = 10, neval_warm = 10000,
                      nitn = 10, neval = 10000,
                      errTol=0.1, maxIter=20,seed=99999,
                      extra_args=list(mu=mu, cov=cov))
  # extra_args are additional arguments needed for myf
  testthat::expect_equal(r_result[1], vegas_result$mean, tolerance = 1e-4)

})

################################################################################
test_that("vegas is wrapped correctly - nunmerical check 2", {
  # check Vegas is working correctly by using it to integrate existing density
  # functions i.e. a dXXX() function in R, and check against the pXXX()
  # function. use arbitrary upper and lower bounds
  # Specific example1 a 5-D mvt
  skip_if_not_installed("mvtnorm")
  skip_on_cran()  # This test will be skipped on CRAN servers
  skip_on_ci()

  # Use library(mvtnorm) first for 3-D Multivariate Normal Density
  library(mvtnorm)
  # use built-in pmvnorm() - distribution function
  lower=rep(-1,5); upper=rep(1,5)
  set.seed(99999) # this uses monte carlo so set for reproducibility
  r_result<-pmvt(lower=lower, upper=upper, sigma=diag(5) )
  print(r_result)
  ## now use vegas to compute same integral
  library(vegasr)
  vegas_initialize()

  # call the built-in density function and use vegas to integrate it
  myf<-function(x,sigma){
    res<-dmvt(x,sigma=sigma,log=FALSE) # need to turn off log
    return(res)
  }

  ## See help page for descriptions of warm and nitn and neval.
  vegas_result<-vegas(f=myf,
                      lower=lower, upper=upper,
                      nitn_warm = 10, neval_warm = 10000,
                      nitn = 10, neval = 10000,
                      errTol=0.001, maxIter=20,seed=99999,
                      extra_args=list(sigma=diag(5)))
  print(vegas_result)
  # extra_args are additional arguments needed for myf
  testthat::expect_equal(r_result[1], vegas_result$mean, tolerance = 1e-4)

})


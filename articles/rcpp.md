# Using Rcpp in Vegasr

### Quickstart

11th-May-update: An example of how to use RcppArmadillo to compute an
integrand, which is typically much faster than using R directly. This
compares the R function and the Rcpp function each passed to vegas.

## Model Formulation

### Data set

``` r
library(vegasr)
vegas_initialize()
#> successfully initialized vegas version: 6.4.1
thedata<-vegasr:::fn_create_data_1(99999) # a list of y and treat as matrices
# this function is in vegasr/R/fn_internal.R
```

``` r
### Now use VEGAS
library(RcppArmadillo)
#> Warning: package 'RcppArmadillo' was built under R version 4.2.3

theta<- matrix(data=rep(0.1,length=4*6), ncol = 6)

vegasr:::fn_log_post_1(theta, thedata$y, thedata$treat,.0, 1.0)
#> [1] -145.933 -145.933 -145.933 -145.933
# this function is in vegasr/R/fn_internal.R

vegasr::arma_fn_log_post_1(theta, thedata$y, thedata$treat,0.0, 1.0)
#>          [,1]
#> [1,] -145.933
#> [2,] -145.933
#> [3,] -145.933
#> [4,] -145.933
# this function is in vegasr/src/fns.cpp
```

``` r
library(vegasr)
# now setup python environment
vegas_initialize() # this needed called once per session after library(vegas)
#> vegas is already initialized
#> NULL
library(tictoc)
#> Warning: package 'tictoc' was built under R version 4.2.3


tic()
result_logEv<-vegasBayesEvidence(f=vegasr:::fn_log_post_1,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=0.1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
#> 113.83 sec elapsed
cat("log evidence = ",result_logEv,"\n")
#> log evidence =  -129.4151


tic()
result_logEv<-vegasBayesEvidence(f=arma_fn_log_post_1,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=0.1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
#> 17.33 sec elapsed
cat("log evidence = ",result_logEv,"\n")
#> log evidence =  -129.4151
```

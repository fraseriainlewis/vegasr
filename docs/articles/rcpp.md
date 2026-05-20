# Using Rcpp in Vegasr

## Quickstart

Using Rcpp libraries to write the log posterior function will almost
surely be faster, perhaps many times faster, than an R function even
when vectorized. Examples of the timing comparison are below.

Example log posterior functions are given below using two Rcpp linear
algebra libraries,
[RcppArmadillo](https://github.com/rcppcore/rcpparmadillo) and
[RcppEigen](https://github.com/RcppCore/RcppEigen). These libraries have
different pros and cons. RcppArmadillo has some built-in parallelization
(via openMP) while RcppEigen does not, but both are highly efficient in
their own way. RcppEigen works well with
[RcppParallel](https://github.com/rcppcore/rcppparallel) which does not
use openMP and can be more problematic with RcppArmadillo.

## Model Formulation

The model implemented here is:

``` math

\begin{aligned}
\mu_0 &\sim \text{Normal}(0, 2.5)\\
\sigma_0 &\sim \text{Half-Normal}(0, 2.5) \\
\mu_1 &\sim \text{Normal}(0, 2.5) \\
\sigma_1 &\sim \text{Half-Normal}(0, 2.5) \\
\beta_0 &\sim \text{Normal}(\mu_0, \sigma_0) \\
\beta_1 &\sim \text{Normal}(\mu_1, \sigma_1) \\
\text{logit(}{p_i}) &= \beta_0 + \beta_1 z_i\quad\quad\quad\quad\enspace\enspace\text{for}\enspace{i=1,\dots,N}  \\
y_i &\sim \text{Bernoulli}(p_i)
\end{aligned}
```

## Data set

The R chunk below creates a simple dataset of three cols:

- a binary response variable y (0/1 = non-responder/responder),
- a (dummy) basket ID variable (=1)
- a binary treatment variable (0/1 = control/test treatment)

The basket ID is currently set fixed at 1, denoting there is only one
basket in this trial, i.e. it’s a classical two arm randomized trial
design.

``` r
library(vegasr)
vegas_initialize()
#> successfully initialized vegas version: 6.4.1
thedata<-vegasr:::fn_create_data_1(99999) # a list of y and treat as matrices
# this function is in vegasr/R/fn_internal.R
```

## Example of calling log posterior functions

These functions are available in the source package in fns_arma.cpp or
fns_eigen.cpp in /src.

``` r
### Call the Rcpp functions to demonstrate correct input and output values

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
# this function is in vegasr/src/fns_arma.cpp

vegasr::eigen_fn_log_post_1(theta, thedata$y, thedata$treat,0.0, 1.0)
#> [1] -145.933 -145.933 -145.933 -145.933
# this function is in vegasr/src/fns_eigen.cpp
```

## Timing comparisons

``` r
library(vegasr)
# now setup python environment
vegas_initialize() # this needed called once per session after library(vegas)
#> vegas is already initialized
#> NULL
library(tictoc)
#> Warning: package 'tictoc' was built under R version 4.5.3

## pure R
tic()
result_logEv<-vegasBayesEvidence(f=vegasr:::fn_log_post_1,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
#> 30.47 sec elapsed
cat("log evidence = ",result_logEv,"\n")
#> log evidence =  -129.4204

## Armadillo
tic()
result_logEv<-vegasBayesEvidence(f=vegasr:::arma_fn_log_post_1,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
#> 5.72 sec elapsed
cat("log evidence = ",result_logEv,"\n")
#> log evidence =  -129.4204

## Eigen
tic()
result_logEv<-vegasBayesEvidence(f=vegasr::eigen_fn_log_post_1,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
#> 5.84 sec elapsed
cat("log evidence = ",result_logEv,"\n")
#> log evidence =  -129.4204

## Eigen with parallel
tic()
result_logEv<-vegasBayesEvidence(f=vegasr::eigen_fn_log_post_1_par,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
#> 1.19 sec elapsed
cat("log evidence = ",result_logEv,"\n")
#> log evidence =  -129.4204
```

## 2. Find Marginal

Example using parallel version of log posterior using Eigen library. See
fns_eigen.cpp in /src in source package.

``` r

tic()
mymarg<-vegasBayesPosterior(f=vegasr::eigen_fn_marg_1_1_par,
                            lower=c(-1,-1,-1,0.0001,0.0001),
                            upper=c(1,1,1,1,1),
                            nitn_warm = 10, neval_warm = 10000,
                            nitn = 10, neval = 10000,
                            errTol=1,maxIter=10,seed=99999,nsearch=10000,
                            log_evidence = result_logEv,
                            extra_args=list(
                              y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.,z=-1.))
cat("Marginal density f(z) at z = -1. = ",mymarg,"\n")
#> Marginal density f(z) at z = -1. =  1.420125
toc()
#> 0.28 sec elapsed

myz<-seq(-2.5,-0.,len=no_den_pts)
tic("") # Start timer with a label
f_z<-rep(0,length(myz));
i<-1;
for(z in myz){
  f_z[i]<-vegasBayesPosterior(f=vegasr::eigen_fn_marg_1_1_par,
                            lower=c(-1,-1,-1,0.0001,0.0001),
                            upper=c(1,1,1,1,1),
                            nitn_warm = 10, neval_warm = 10000,
                            nitn = 10, neval = 10000,
                            errTol=1,maxIter=10,seed=99999,nsearch=10000,
                            log_evidence = result_logEv,
                            extra_args=list(
                              y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.,z=z))
  i<-i+1
}

toc() # Stops timer and prints
#> : 20.61 sec elapsed
```

``` r
# 3. Display the plot
# plotting code above in hidden chunk
if(!nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_"))){
print(p)
}
```

![](rcpp_files/figure-html/plot_marg2-1.png)

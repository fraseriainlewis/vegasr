
<!-- README.md is generated from README.Rmd. Please edit that file -->

# vegasr <img src="man/figures/logo.png" align="right" width="200"/>

<!-- badges: start -->

<!-- badges: end -->

Vegasr is an R library for **multi-dimensional numerical integration**.
It is a thin wrapper around the Python
[vegas](https://vegas.readthedocs.io/en/latest/index.html) library which
was developed is maintained by the author of the vegas algorithm, G. P.
Lepage. The purpose of this library is to allow ready access to the
latest [vegas+](https://arxiv.org/abs/2009.05112) algorithm from R.

The vegas algorithm performs efficient Monte Carlo integration for
integrals of modest to high dimension where cubature or hcubature
methods are computationally impractical. The
[vegas+](https://arxiv.org/abs/2009.05112) algorithm is a more efficient
version of the original and widely used vegas algorithm. For the
original vegas article see J. Comput. Phys. 27 (1978) 192 and for later
vegas+ see J. Comput. Phys. 439 (2021) 110386).

[Getting Started
vignette](https://fraseriainlewis.github.io/vegasr/articles/introduction.html)

## Installation

This library uses [reticulate](https://rstudio.github.io/reticulate/) to
pass objects back and forth to Python vegas and so Python must be
available. There is a separate install script which deals with the
necessary Python dependencies. The vegasr GitHub repo is
[here](https://github.com/fraseriainlewis/vegasr).

You can install the development version of vegasr from
[GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("fraseriainlewis/vegasr")
vegas_install() # this installs necessary python libraries
```

## Example - Integrate over Multivariate Normal Density

This is a basic example to show the integrand function structure and
core functionality. The integrand is a 3-D multivariate normal density.
R has an existing function in the
[mvtnorm](https://cran.r-project.org/web/packages/mvtnorm/index.html)
library for efficiently computing this multivariate integral, .

``` r
# Use library(mvtnorm) first for 3-D Multivariate Normal Density
library(mvtnorm) 
mu<-matrix(c(0.5, -0.2, 0.1),nrow=1) # note matrix
cov<-matrix(data=c(
                 1.0, 0.5, 0.2,
                 0.5, 1.2, 0.3,
                 0.2, 0.3, 0.8),ncol=3,byrow=FALSE)

# use built-in pmvnorm()
lower=c(-0.5,-0.5,-0.5); upper=c(2.,1.,3.)
res_builtin<-pmvnorm(lower=lower,upper=upper,
                     mean=as.numeric(mu), # coercion as needs vector
                     sigma=cov)
print(res_builtin)
#> [1] 0.3103351
#> attr(,"error")
#> [1] 3.768512e-05
#> attr(,"msg")
#> [1] "Normal Completion"

## now use vegas to compute same integral
library(vegasr)
## Important - run next line in each R session to ensure python is ready
vegas_initialize()
#> successfully initialized vegas version: 6.4.1

# the integrand function MUST take a matrix of dimension [BATCH,M] as first 
# argument and return a vector of length M (or 1-row matrix). 
# Any number of other additional *named* arguments are allowed and should all 
# be matrices. See ?vegas_integrate for more details.
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
                        errTol=0.1,maxIter=20,seed=99999,
                        extra_args=list(mu=mu,cov=cov)) # these are additional arguments needed for myf
print(vegas_result)
#> $mean
#> [1] 0.3103397
#> 
#> $error
#> [1] 4.470286e-05
#> 
#> $metTolerance
#> [1] 1
```

**Remarks**:

- Numerical accuracy is controlled by `errTol`, the target percentage
  error, e.g., 0.1 above is 0.1%

- Error is technically the estimated standard deviation of the mean
  estimate of the value of the integrand.

- Parameters `nitn_warm, neval_warm, nitn, neval` control the
  computational accuracy and in effect add more and more Monte Carlo
  samples until the desired accuracy is achieved. The final integrand
  value and error are weighted across all samples.

- An adaptive grid is used and very early samples are discarded (the
  warm-up period) as these may be noisy until the grid has adapted and
  including may result in a larger than necessary error estimate.

- An important remark on the function passed to `vegas()` is that if
  using an existing R function then it may be best to wrap it into a
  simpler function template. A concrete example of this issue is using
  `f=dmvnorm` in the above call to `vegas()` does not work and throws
  and error about *esoteric Python-incompatible constructs*

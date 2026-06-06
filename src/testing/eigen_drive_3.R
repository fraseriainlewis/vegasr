#library(RcppArmadillo)
library(RcppEigen)
#library(RcppParallel)
#library(tictoc)
## provides dmvnorm_aram(x,my,cov)

#Rcpp::sourceCpp("src/testing/eigen_v1.cpp")
Rcpp::sourceCpp("src/testing/fns_imp_eigen.cpp")

#mylist<-list(a=matrix(data=rnorm(3*4),ncol=1),b=matrix(data=rnorm(2*4),ncol=1))

#eigen_grid(mylist)

library(vegasr)
## Important - run next line in each R session to ensure python is ready
vegas_initialize()

thedata<-vegasr:::fn_create_data_1(99999)

###### 1. compute a good grid
# 1.a get shiftby to help numerical robustness
set.seed(9999)
start<-c(-1,-1,-1,-1,0.0001,0.0001)
stop<-c(1,1,1,1,1,1)
nsearch=100000
searchPts <- mapply(seq, from = start, to = stop,
                    MoreArgs = list(length.out = nsearch))
for(i in 1:length(start)){searchPts[,i]<-searchPts[sample(1:nrow(searchPts)),i]}
mymax<-max(vegasr:::fn_log_post_1(searchPts,y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
# mymax on LOG scale

# 1b. compute the integrand - note log
vegas_result<-vegasip(f=vegasr:::eigen_fn_log_post_1_par,
                      lower=c(-1,-1,-1,-1,0.0001,0.0001), upper=c(1,1,1,1,1,1),
                      nitn_warm = 10, neval_warm = 100000,
                      nitn = 10, neval = 100000,
                      errTol=0.1, maxIter=10,seed=99999,adapt=TRUE,
                      extra_args=list(y=thedata$y,treat=thedata$treat,shiftby=mymax,uselog=0.))
vegas_result$metTolerance

# 2. generate random sample from p(y), y is internal vegas variable and generate x and jac from grid
# and generate weights
#set.seed(9999)
y_rv<-matrix(data=runif(500000*length(vegas_result$x_grid)),ncol=length(vegas_result$x_grid))

# get X and Jac
res<-eigen_gridM(vegas_result$x_grid,
                 y_rv# this is just runif(), using runif from python currently
                 )
myX<-res[[1]]
myJac<-res[[2]]
# now compute f(x) and weights - note using shiftby and uselog, so need to revert those
f_x<-vegasr:::eigen_fn_log_post_1_par(myX,y=thedata$y,treat=thedata$treat,shiftby=mymax,uselog=1.)

wgts<-exp(mymax+f_x)*myJac # need to add back in shiftby and then exponentiation
#wgts<-f_x*myJac
wgts<-pmin(pmax(wgts, 0), 1) # in case weight -ve, should not happen
wgts_std <- wgts/sum(wgts) # standardize

#wgts_std[422057]<-mean(wgts_std[-422057]);
#wgts_std<-pmin(pmax(wgts_std, 0), 1) # in case weight -ve, should not happen
#wgts_std <- wgts_std/sum(wgts_std) # standardize

cat("ESS = ",1/sum(wgts_std^2),"\n")

# 3. generate sampling from X using
my_IP_sample<-sample.int(nrow(myX),size=50000,replace=TRUE,prob=wgts_std)
myX2<-myX[my_IP_sample,]

i<-1
plot(plot_data[,1],plot_data[,2],type="l")
lines(density(myVar<-myX2[,i]/(1-myX2[,i]^2)),col="red")
print(quantile(myVar,c(0.025,0.975)))

i<-2
plot(density(myX[,i]/(1-myX[,i]^2)),col="red")


newX<-(-1+sqrt(1+4*myX_sample[,1]^2))/(2*myX_sample[,1])

plot( (-1+sqrt(1+4*myX[,1]^2))/(2*myX[,1]))


plot(density(myX_sample[,1]),col="red")

plot(density(newX),col="red")

# wgt and x.    wgt = f(x)*jac
# sample x with this weight
# for integral sampling






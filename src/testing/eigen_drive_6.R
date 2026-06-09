#library(RcppArmadillo)
library(RcppEigen)
#library(RcppParallel)
#library(tictoc)
## provides dmvnorm_aram(x,my,cov)

#Rcpp::sourceCpp("src/testing/eigen_v1.cpp")
#Rcpp::sourceCpp("src/fns_imp_eigen.cpp")

#mylist<-list(a=matrix(data=rnorm(3*4),ncol=1),b=matrix(data=rnorm(2*4),ncol=1))

#eigen_grid(mylist)

library(vegasr)
## Important - run next line in each R session to ensure python is ready
vegas_initialize()

thedata<-vegasr:::fn_create_data_5(99999)

#############################################################################################
# 1. Run vegas to compute integrand - log evidence and then manually use importance sampling
# from the same grid and compare result. uses different random samples but same grid (weights)
# so should be same up to monte-carlo error.

K <- length(unique(thedata$basket))
lower <- c(rep(-0.9999, 2*K), -0.9999, -0.9999, 1e-4, 1e-4)
upper <- c(rep( 0.9999, 2*K),  0.9999,  0.9999,   0.9999,   0.9999)

result <- vegasBayesEvidence(
  f = vegasr::eigen_fn_log_post_2_par,
  lower = lower, upper = upper,
  nitn_warm = 10, neval_warm = 100000,
  nitn = 10, neval = 20000,
  errTol = 1, maxIter = 10, seed = 999199, nsearch = 100000,
  extra_args=list(
    y=thedata$y,
    treat=thedata$treat,
    basket = thedata$basket,
    shiftby=0,uselog=1.)
)

cat("log evidence direct from vegas = ",result$log_evidence,"\n")

#
f_x<-vegasr:::eigen_fn_log_post_2_par(result$x_vals, # these are random samples from the final grid
                                       y=thedata$y,
                                       treat=thedata$treat,
                                       basket = thedata$basket,
                                       shiftby=result$shiftby, # need to use same offset as above
                                       uselog=1. # use log as just function eval
                                      )

# result$pwgts - these are sampling weights - the internal jacobian and volume adjustment
# this is usual importance sampling for integrand - function value times weight
# this weight has n in it so it's sum below not mean (but conceptually it's mean)
log_ev2<-log(sum(exp(result$shiftby+f_x)*result$pwgts)) # adjust by offset,

cat("log evidence resampled from vegas = ",log_ev2,"\n")


#############################################################################################
# 2. Use the final grid to estimate marginal quantiles

# generate
y_rv<-matrix(data=runif(500000*length(result$x_grid)),ncol=length(result$x_grid))

# get X and Jac
res<-vegasr:::eigen_gridM(result$x_grid,
                 y_rv# this is just runif(), using runif from python currently
)
myX<-res[[1]]
myJac<-res[[2]]
# now compute f(x) and weights - note using shiftby and uselog, so need to revert those
f_x<-vegasr:::eigen_fn_log_post_2_par(myX,
                                      y=thedata$y,
                                      treat=thedata$treat,
                                      basket = thedata$basket,
                                      shiftby=result$shiftby,uselog=1.)


wgts<-exp(result$shiftby+f_x)*myJac # need to add back in shiftby and then exponentiation
wgts_std <- wgts/sum(wgts) # standardize

## from here we have all the info we need to compute all the quantiles for all variables with no more computation
# consider X1
myorder1<-order(myX[,1]) # sort asc idx
X1o<-myX[myorder1,1] # sort X1 asc
w1<-wgts_std[myorder1] # order weights
w1b<-cumsum(w1); # cumsum weights
p25X<-X1o[(which(w1b>0.025)[1])] # get 2.5% on x scale -1,+!
p25Z<-p25X/(1-p25X^2) # transform back to -inf, + inf

cat("Variable X1 = 2.5% ",p25Z,"\n")

# 3. generate sampling from X using
my_IP_sample<-sample.int(nrow(myX),size=1000000,replace=TRUE,prob=wgts_std)
myX2<-myX[my_IP_sample,]



#wgts<-f_x*myJac
#cliptop<-which(wgts>quantile(wgts,0.9999)) #
desc<-summary(wgts)
maxwgt<-desc["3rd Qu."]+10*(desc["3rd Qu."]-desc["1st Qu."])
#wgts2<-pmin(pmax(wgts, 0), quantile(wgts,0.9999)) # clip negative and top 0.01%
dropme<-which(wgts>maxwgt)
wgts2<-wgts[-dropme]
myX<-myX[-dropme,]
#plot(wgts2)
#wgts2<-wgts
wgts_std <- wgts2/sum(wgts2) # standardize

#wgts_std[422057]<-mean(wgts_std[-422057]);
#wgts_std<-pmin(pmax(wgts_std, 0), 1) # in case weight -ve, should not happen
#wgts_std <- wgts_std/sum(wgts_std) # standardize

cat("ESS = ",1/sum(wgts_std^2),"\n")

# 3. generate sampling from X using
my_IP_sample<-sample.int(nrow(myX),size=1000000,replace=TRUE,prob=wgts_std)
myX2<-myX[my_IP_sample,]

i<-1
#plot(plot_data[,1],plot_data[,2],type="l")
#lines(density(myVar<-myX2[,i]/(1-myX2[,i]^2)),col="red")

plot(density(myVar<-myX2[,i]/(1-myX2[,i]^2),adjust=3),col="red")
myVar<-myX2[,i]/(1-myX2[,i]^2)
print(quantile(myVar,c(0.025)))
print(quantile(myVar,c(0.975)))
i<-2
plot(density(myVar<-myX2[,i]/(1-myX2[,i]^2)),col="blue")
i<-5
plot(density(myVar<-myX2[,i]/(1-myX2[,i]^2)),col="blue")


#### quantiles
X1<-myX[,1]
X1.srt<-sort(X1)
wgts_std.srt<-wgts_std[order(X1)]
idx<-which(cumsum(wgts_std.srt)>0.975)[1]
a<-X1.srt[idx];
a/(1-a^2)



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






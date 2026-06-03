####################### SETUP
## now use vegas to compute same integral
library(vegasr)
## Important - run next line in each R session to ensure python is ready
vegas_initialize()

myf<-function(x,mu,cov){
  res<-dmvnorm(x,
               mean = mu, # this is a 1-row matrix, dmvnorm accepts this
               sigma=cov)
  return(res)
}

mu<-matrix(c(0.5, -0.2, 0.1),nrow=1) # note matrix
cov<-matrix(data=c(
  1.0, 0.5, 0.2,
  0.5, 1.2, 0.3,
  0.2, 0.3, 0.8),ncol=3,byrow=FALSE)

## See help page for descriptions of warm and nitn and neval.
vegas_result<-vegasip(f=myf,
                    lower=c(-0.5,-0.5,-0.5), upper=c(2.,1.,3.),
                    nitn_warm = 10, neval_warm = 10000,
                    nitn = 10, neval = 10000,
                    errTol=0.1, maxIter=20,seed=99999,adapt=TRUE,
                    extra_args=list(mu=mu, cov=cov))
# extra_args are additional arguments needed for myf

###### COMPUTE GRID JACOBIAN and weights now check that the jac

grid<-vegas_result$x_grid;
y<-vegas_result$y_1 # random on [0,1]^3
x<-vegas_result$x_1

y11<-y[1,1] # which k bins is myy in
y12<-y[1,2]
y13<-y[1,3]
#now get x
inc1<-diff(grid[[1]])
ninc1<-length(inc1)
inc2<-diff(grid[[2]])
ninc2<-length(inc2)
inc3<-diff(grid[[3]])
ninc3<-length(inc3)

k11<-floor(y11*ninc1)+1 # logic y has equal increments from 0 through 1
k12<-floor(y12*ninc2)+1 # logic ctd , y close to 0 = 0, y close = 1 ninc1-1, floor does this
k13<-floor(y13*ninc3)+1 # +1 because R is indexes from 1 not 0, otherwise remove the +1

#grid[d, k] + (y_d − k/ninc) × ninc × inc[d, k]
x11<-grid[[1]][k11] + (y11 - (k11-1)/ninc1)*ninc1*inc1[k11] # logic y and x map into same k index so just need to find left hand edge
x12<-grid[[2]][k12] + (y12 - (k12-1)/ninc2)*ninc2*inc2[k12] # of boundary, i.e. grid[[1]][k11] and then find out proportion of current bin used
x13<-grid[[3]][k13] + (y13 - (k13-1)/ninc3)*ninc3*inc3[k13] # i.e. y is in [ k/ninc, (k+1)/ninc ] on 0-index, -(k11-1)/ninc then leaves just
                                                       # the delta above the edge of the bins, then scale this by jacobian (width_x_bin/width_y_bin)
                                                       # i.e. inc1[k11] is width of x-bin and 1/ninc1 is width of y bin

# compare
print(myx<-c(x11,x12,x13))
print(x[1,])

# jacobian is ninc*incr
myjac<-ninc1*inc1[k11] * ninc2*inc2[k12] * ninc3*inc3[k13]
print(myjac)
print(vegas_result$jac1_10[1])
print(vegas_result$jac2_10[1])

# to get actual weights need to multiply by f(x)
myXX<-matrix(data=myx,nrow=1,byrow=TRUE)

# get weight for X
myf(myXX,mu,cov)*myjac



















#x_actual<-vegas_result$x_1

#ninc = amap.ninc
#inc  = amap.inc
#ndim = amap.dim

#jac3 = 1.0#np.ones(y.shape[0])
#for d in range(ndim):
#  k = np.floor(y[:, d] * ninc[d]).astype(int)
#  k = np.clip(k, 0, np.array(ninc[d]) - 1)
# jac3 *= ninc[d] * np.array(inc)[d, k]
# print(jac3)





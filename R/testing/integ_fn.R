## create R function for using with Vegas

#// [[Rcpp::export]]
# NumericVector fast_integrand(NumericMatrix x) {
#     // This 'x' is just a pointer to Python memory.
#     // No R matrices were created or destroyed in the making of this calculation.
#     // ... your math here ...
#   }



library(mvtnorm)

myf<-function(x){
   print(dim(x))
   x <- as.matrix(x)
   #hard code mean and var for ease so no params to pass
   res<-dmvnorm(x, mean =
                  matrix(c(0.5, -0.2, 0.1),nrow=1),
          sigma = matrix(data=c(
    1.0, 0.5, 0.2,
    0.5, 1.2, 0.3,
    0.2, 0.3, 0.8),ncol=3,byrow=FALSE))
   return(res)

}

# batching - using (BATCH,dim) form
a<-matrix(data=c(0.1,0.1,0.2,
                 0.1,0.1,0.2),ncol=3,byrow=TRUE)
myf(a)

library(reticulate)
# send to python
myfref<-reticulate::r_to_py(myf)
main <- reticulate::import_main(convert = FALSE)
main$r_func <- myf



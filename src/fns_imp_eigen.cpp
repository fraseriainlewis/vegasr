#define EIGEN_DONT_PARALLELIZE
#include <RcppEigen.h>
#include <RcppParallel.h>
#include <cmath>

// [[Rcpp::depends(RcppEigen, RcppParallel)]]
// [[Rcpp::plugins(cpp14)]]

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::VectorXi;

// [[Rcpp::export]]
Rcpp::List diff_list(const Rcpp::List& grid) {
  int n = grid.size();
  Rcpp::List result(n);

  for (int i = 0; i < n; i++) {
    const Eigen::Map<Eigen::VectorXd> v(
        Rcpp::as<Eigen::Map<Eigen::VectorXd>>(grid[i])
    );
    // Equivalent of R's diff(): consecutive differences
    Eigen::VectorXd d = v.tail(v.size() - 1) - v.head(v.size() - 1);
    result[i] = d;
  }
  return result;
}


// [[Rcpp::export]]
Eigen::VectorXd len_list(const Rcpp::List& grid) {
  int n = grid.size();
  Eigen::VectorXd len(n);

  for (int i = 0; i < n; i++) {
    const Eigen::Map<Eigen::VectorXd> v(
        Rcpp::as<Eigen::Map<Eigen::VectorXd>>(grid[i])
    );
    // Equivalent of R's diff(): consecutive differences
   len(i)=v.size();
  }
  return len;
}





// [[Rcpp::export]]
void eigen_grid(const Rcpp::List& xgrid, const VectorXd& y) {


  Rcpp::Rcout <<"y="<<y<<std::endl;
  //Rcpp::Rcout <<"m2="<<diff_list(xgrid)<<std::endl;

  /*y1<-c(y[1,1],y[1,2],y[1,3]) # which k bins is myy in

   inc<-lapply(grid,diff) # increments in each grid
#inc2<-diff_list(grid)
   ninc<-as.numeric(lapply(inc,length)) # number of bins
#ninc2<-len_list(inc)
   k<-floor(y1*ninc)+1 # find the bind in each grid

# find x using bin and scaling
   x1<-rep(0,3)
   for(i in 1:3){
   x1[i]<-grid[[i]][k[i]] + (y1[i] - (k[i]-1)/ninc[i])*ninc[i]*inc[[i]][k[i]]}
   */

  //return(diff_list(xgrid));

  int m = xgrid.size();
  Rcpp::List inc = diff_list(xgrid);
  Eigen::VectorXd ninc = len_list(inc);

  Rcpp::Rcout<<"here is ninc "<<ninc<<std::endl;
  //Rcpp::Rcout<<"here is inc[[1]]"<<inc[0]<<std::endl;

  Eigen::VectorXd k = (y.array()*ninc.array()).array().floor();
  Eigen::VectorXd x(m);
  Eigen::VectorXd jac(1); jac(0)=1.0;

  Rcpp::Rcout<<"k ="<<k<<std::endl;

  //double tmp=0.0;
  for (auto i = 0; i < m; ++i) {
    const Eigen::Map<Eigen::VectorXd> grid_i(
      Rcpp::as<Eigen::Map<Eigen::VectorXd>>(xgrid[i]));

    const Eigen::Map<Eigen::VectorXd> inc_i(
        Rcpp::as<Eigen::Map<Eigen::VectorXd>>(inc[i]));

    x(i)=grid_i((int)k(i)) + (y(i)-k(i)/ninc(i))*ninc(i)*inc_i(int(k(i)));

    Rcpp::Rcout<<"x[i]"<<x(i)<<std::endl;
    //Rcpp::Rcout<<"v(k(i))"<<v((int)k(i))<<std::endl;

    jac(0)*=ninc(i)*inc_i(int(k(i)));
  }

  Rcpp::Rcout<<"jac"<<jac(0)<<std::endl;

  // #jaco
  // i<-1
  // ninc[i]*inc[[i]][k[i]]*
  // ninc[i+1]*inc[[i+1]][k[i+1]]*
  // ninc[i+2]*inc[[i+2]][k[i+2]]


  // find
  //Rcpp::Rcout <<"m="<<xgrid[0]<<std::endl;

  //for (int i = 0; i < n; i++) {
  // Zero-copy map by index
  //const Eigen::Map<Eigen::MatrixXd> A(
  //    Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(xgrid[0])
  //);
  //}

  //Rcpp::Rcout <<"m="<<A<<std::endl;
  //return(A);
}


// [[Rcpp::export]]
Rcpp::List eigen_gridM(const Rcpp::List& xgrid, const Eigen::MatrixXd& y) {

  int m = xgrid.size();
  int r = y.rows();
  int c = y.cols();

  Rcpp::List result(2);

  //Rcpp::Rcout<<"y maxrix"<<y<<std::endl;

  Rcpp::List inc = diff_list(xgrid);
  Eigen::VectorXd ninc = len_list(inc);

  Eigen::MatrixXd X(r,c);
  Eigen::VectorXd thejac(r);

  for(auto j=0;j<r;j++){// for each y vector of values, vegas internal scale vector

    Eigen::VectorXd k = (y.row(j).transpose().array() * ninc.array()).floor().matrix();
    Eigen::VectorXd x(m);
    Eigen::VectorXd curjac(1); curjac(0)=1.0;
    //Rcpp::Rcout<<"k="<<k<<std::endl;

  for (auto i = 0; i < m; ++i) {// for each dimension within y vector
    const Eigen::Map<Eigen::VectorXd> grid_i(
        Rcpp::as<Eigen::Map<Eigen::VectorXd>>(xgrid[i]));

    const Eigen::Map<Eigen::VectorXd> inc_i(
        Rcpp::as<Eigen::Map<Eigen::VectorXd>>(inc[i]));

    x(i)=grid_i((int)k(i)) + (y(j,i)-k(i)/ninc(i))*ninc(i)*inc_i(int(k(i)));

    //Rcpp::Rcout<<"x[i]"<<x(i)<<std::endl;

    curjac(0)*=ninc(i)*inc_i(int(k(i)));
  }

  //Rcpp::Rcout<<"jac"<<curjac(0)<<std::endl;
  thejac(j)=curjac(0);
  X.row(j) = x.transpose();
  }

  result[0]=X;
  result[1]=thejac;




  return (result);

}





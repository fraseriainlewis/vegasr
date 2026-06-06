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
Rcpp::List eigen_gridM(const Rcpp::List& xgrid, const MatrixXd& y) {

  int m = xgrid.size();
  int r = y.rows();
  int c = y.cols();

  Rcpp::List result(2);

  //Rcpp::Rcout<<"y maxrix"<<y<<std::endl;

  Rcpp::List inc = diff_list(xgrid);
  Eigen::VectorXd ninc = len_list(inc);

  Eigen::MatrixXd X(r,c);
  VectorXd thejac(r);

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




// not exported
VectorXd eigen_norm_logpdf(const VectorXd& x, const VectorXd& loc, const VectorXd& scale) {
  ArrayXd x_arr = x.array();
  ArrayXd loc_arr = loc.array();
  ArrayXd scale_arr = scale.array();

  ArrayXd log_var = -0.5 * ((x_arr - loc_arr) / scale_arr).square();
  ArrayXd log_const = -scale_arr.log() - 0.5 * std::log(2.0 * M_PI);

  return (log_var + log_const).matrix();
}

// not exported
MatrixXd eigen_norm_logpdf(const MatrixXd& x, const VectorXd& loc, const VectorXd& scale) {
  ArrayXXd x_arr    = x.array();
  ArrayXd  loc_arr  = loc.array();
  ArrayXd  scale_arr = scale.array();

  ArrayXXd scaled    = (x_arr.colwise() - loc_arr).colwise() / scale_arr;
  ArrayXXd log_var   = -0.5 * scaled.square();
  ArrayXXd log_const = (-scale_arr.log() - 0.5 * std::log(2.0 * M_PI)).matrix()
                                  .replicate(1, x.cols()).array();

  return (log_var + log_const).matrix();
}


// not exported
VectorXd eigen_norm_logpdf(const VectorXd& x, double loc, double scale) {
  ArrayXd x_arr = x.array();

  ArrayXd log_var = -0.5 * ((x_arr - loc) / scale).square();
  double log_const = -std::log(scale) - 0.5 * std::log(2.0 * M_PI);

  return (log_var + log_const).matrix();
}

// not exported
VectorXd eigen_half_norm_logpdf(const VectorXd& x, double sigma) {
  double log_const = 0.5 * (std::log(2.0) - std::log(M_PI)) - std::log(sigma);
  ArrayXd log_exp = -(x.array().square() / (2.0 * sigma * sigma));

  return (log_const + log_exp).matrix();
}

// Helper: Scalar Normal Log-PDF
inline double norm_logpdf_scalar(double x, double mu, double sigma) {
  return -std::log(sigma) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow((x - mu) / sigma, 2);
}

// Helper: Scalar Half-Normal Log-PDF
inline double half_norm_logpdf_scalar(double x, double sigma) {
  return 0.5 * (std::log(2.0) - std::log(M_PI)) - std::log(sigma) - (x * x / (2.0 * sigma * sigma));
}



// [[Rcpp::export]]
Eigen::VectorXd eigen_fn_log_post_11(const Eigen::MatrixXd& theta,
                                    const Eigen::VectorXd& y,
                                    const Eigen::VectorXd& treat,
                                    double shiftby, double uselog){

  int n_rows = theta.rows();

  // Clamp values
  ArrayXd theta0 = theta.col(0).array().max(-0.9999).min(0.9999);
  ArrayXd theta1 = theta.col(1).array().max(-0.9999).min(0.9999);
  ArrayXd theta2 = theta.col(2).array().max(-0.9999).min(0.9999);
  ArrayXd theta3 = theta.col(3).array().max(-0.9999).min(0.9999);
  ArrayXd theta4 = theta.col(4).array().max(0.0001).min(0.9999);
  ArrayXd theta5 = theta.col(5).array().max(0.0001).min(0.9999);


  // Variable transformations
  VectorXd a0 = theta0;
  VectorXd a1 = theta1;
  VectorXd mu0 = theta2 ;
  VectorXd mu1 = theta3 ;
  VectorXd sigma0 = theta4 ;
  VectorXd sigma1 = theta5 ;

  // eta calculation: treat * a1.t() + a0.t()
  // treat is (N x 1), a1 is (BATCH x 1). result should be (N x BATCH)
  // We add a0.transpose() (1 x BATCH) to each row of (N x BATCH) matrix
  MatrixXd eta = (treat * a1.transpose()).rowwise() + a0.transpose();

  // term1 = y * eta (element-wise on y broadcasted across cols)
  MatrixXd term1 = eta.array().colwise() * y.array();

  // term2 = log(1 + exp(eta))
  MatrixXd term2 = (1.0 + eta.array().exp()).log();

  // logL = sum(term1 - term2, axis=0) -> returns row vector (1 x BATCH)
  Eigen::RowVectorXd logL = (term1 - term2).colwise().sum();

  // Prior calculations
  VectorXd prior_a0 = eigen_norm_logpdf(a0, mu0, sigma0);
  VectorXd prior_a1 = eigen_norm_logpdf(a1, mu1, sigma1);
  VectorXd prior_mu0 = eigen_norm_logpdf(mu0, 0.0, 2.5);
  VectorXd prior_mu1 = eigen_norm_logpdf(mu1, 0.0, 2.5);
  VectorXd prior_sigma0 = eigen_half_norm_logpdf(sigma0, 2.5);
  VectorXd prior_sigma1 = eigen_half_norm_logpdf(sigma1, 2.5);

  // Final log Density
  VectorXd logDens = logL.transpose() + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 + prior_sigma0 + prior_sigma1;

  // Final log Posterior
  VectorXd logPost = logDens;

  if (uselog == 1.0) {
    return (logPost.array() - shiftby).matrix();
  } else {
    return (logPost.array() - shiftby).exp().matrix();
  }
}





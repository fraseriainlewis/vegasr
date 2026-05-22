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

// not exported
VectorXd eigen_norm_logpdf(const VectorXd& x, const VectorXd& loc, const VectorXd& scale) {
  ArrayXd x_arr = x.array();
  ArrayXd loc_arr = loc.array();
  ArrayXd scale_arr = scale.array();

  ArrayXd log_var = -0.5 * ((x_arr - loc_arr) / scale_arr).square();
  ArrayXd log_const = -scale_arr.log() - 0.5 * std::log(2.0 * M_PI);

  return (log_var + log_const).matrix();
}

// [[Rcpp::export]]
MatrixXd eigen_norm2_logpdf(const MatrixXd& x, const VectorXd& loc, const VectorXd& scale) {
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

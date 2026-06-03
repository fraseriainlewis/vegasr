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

// not exported
MatrixXd eigen_grid(const Rcpp::List& xgrid, const MatrixXd& y) {


    const Eigen::Map<Eigen::MatrixXd> A(
      Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(input_list)
    )

    ArrayXd x_arr = x.array();
    ArrayXd loc_arr = loc.array();
    ArrayXd scale_arr = scale.array();

    ArrayXd log_var = -0.5 * ((x_arr - loc_arr) / scale_arr).square();
    ArrayXd log_const = -scale_arr.log() - 0.5 * std::log(2.0 * M_PI);

    return (log_var + log_const).matrix();
}


// [[Rcpp::export]]
Eigen::VectorXd eigen_fn_log_post_1(const Eigen::MatrixXd& theta,
                                    const Eigen::VectorXd& y,
                                    const Eigen::VectorXd& treat,
                                    double shiftby, double uselog){



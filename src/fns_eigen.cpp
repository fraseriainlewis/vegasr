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

// ---------------------------------------------------------------------------
//' @title Posterior Density Function using RcppEigen - Example 1
//' @name eigen_fn_log_post_1
//' @aliases eigen_fn_log_post_1
//' @description An example showing how to write a function for use with \code{\link{vegasBayesEvidence}} for
//' Bayesian computation using the RcppEigen library
//' This example function describes a simple Bayesian hierarchical model comprising of a logistic regression with
//' intercept and single binary covariate for treatment effect each with a hierarchical prior.
//' This has six parameters in total. See \code{vignette("rcpp", package = "vegasr")} for Rcpp details.
//'
//' @details The is an example function written using RcppEigen and has same functionality as the R function
//' \code{\link{fn_log_post_1}}. It uses a transformation so the density
//' can be integrated across the full domain of each parameter, i.e. the density includes a Jacobian
//'  See \code{vignette("rcpp", package = "vegasr")} for more details. Several helper function are required
//'  specifically normal and half-normal densities are also written in RcppEigen. Use Rcpp::sourceCpp()
//'  or similar to run the functions separately. They are in the fns_eigen.cpp file in the source package.
//'
//' @param theta pass a numerical R matrix of dimension Batch x M, where M is number of parameters, here M=6
//' Batch can be any positive integer
//' @param y a numeric R matrix of dimension N x 1, this is the response variable and should be 1.0 or 0.0
//' entries only
//' @param treat a numeric R matrix of dimension N x 1, this is the response variable and should be 1.0 or 0.0
//' entries only
//' @param shiftby a numerical scalar used to help avoid underflow. Used in \code{\link{vegasBayesEvidence}}
//' @param uselog a numerical flag value takes either 1.0 or 0.0 and used to return either log or real scale
//' value. Used in \code{\link{vegasBayesEvidence}}
//' @export
// Define log posterior in RcppEigen including change of variables
// [[Rcpp::export]]
Eigen::VectorXd eigen_fn_log_post_1(const Eigen::MatrixXd& theta,
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

    // Jacobian calculation using Array operations
    ArrayXd jacobianL =
        ((1.0 + theta0.square()).log() - 2.0 * (1.0 - theta0.square()).log()) +
        ((1.0 + theta1.square()).log() - 2.0 * (1.0 - theta1.square()).log()) +
        ((1.0 + theta2.square()).log() - 2.0 * (1.0 - theta2.square()).log()) +
        ((1.0 + theta3.square()).log() - 2.0 * (1.0 - theta3.square()).log()) +
        ((1.0 + theta4.square()).log() - 2.0 * (1.0 - theta4.square()).log()) +
        ((1.0 + theta5.square()).log() - 2.0 * (1.0 - theta5.square()).log());

    // Variable transformations
    VectorXd a0 = (theta0 / (1.0 - theta0.square())).matrix();
    VectorXd a1 = (theta1 / (1.0 - theta1.square())).matrix();
    VectorXd mu0 = (theta2 / (1.0 - theta2.square())).matrix();
    VectorXd mu1 = (theta3 / (1.0 - theta3.square())).matrix();
    VectorXd sigma0 = (theta4 / (1.0 - theta4.square())).matrix();
    VectorXd sigma1 = (theta5 / (1.0 - theta5.square())).matrix();

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
    VectorXd logPost = logDens + jacobianL.matrix();

    if (uselog == 1.0) {
        return (logPost.array() - shiftby).matrix();
    } else {
        return (logPost.array() - shiftby).exp().matrix();
    }
}

// Worker for parallelizing the log-posterior calculation over the batch (theta rows)
struct LogPostWorker : public RcppParallel::Worker {
  const Eigen::MatrixXd& theta;
  const Eigen::VectorXd& y;
  const Eigen::VectorXd& treat;
  Eigen::VectorXd& output;

  LogPostWorker(const Eigen::MatrixXd& theta, const Eigen::VectorXd& y,
                const Eigen::VectorXd& treat, Eigen::VectorXd& output)
    : theta(theta), y(y), treat(treat), output(output) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      // 1. Clamping and Transformations
      double t0 = std::max(-0.9999, std::min(0.9999, theta(i, 0)));
      double t1 = std::max(-0.9999, std::min(0.9999, theta(i, 1)));
      double t2 = std::max(-0.9999, std::min(0.9999, theta(i, 2)));
      double t3 = std::max(-0.9999, std::min(0.9999, theta(i, 3)));
      double t4 = std::max(0.0001, std::min(0.9999, theta(i, 4)));
      double t5 = std::max(0.0001, std::min(0.9999, theta(i, 5)));

      // Jacobian
      double jac = (std::log(1.0 + t0*t0) - 2.0 * std::log(1.0 - t0*t0)) +
        (std::log(1.0 + t1*t1) - 2.0 * std::log(1.0 - t1*t1)) +
        (std::log(1.0 + t2*t2) - 2.0 * std::log(1.0 - t2*t2)) +
        (std::log(1.0 + t3*t3) - 2.0 * std::log(1.0 - t3*t3)) +
        (std::log(1.0 + t4*t4) - 2.0 * std::log(1.0 - t4*t4)) +
        (std::log(1.0 + t5*t5) - 2.0 * std::log(1.0 - t5*t5));

      double a0 = t0 / (1.0 - t0*t0);
      double a1 = t1 / (1.0 - t1*t1);
      double mu0 = t2 / (1.0 - t2*t2);
      double mu1 = t3 / (1.0 - t3*t3);
      double sigma0 = t4 / (1.0 - t4*t4);
      double sigma1 = t5 / (1.0 - t5*t5);

      // 2. Likelihood Calculation (Vectorized for the current row)
      Eigen::ArrayXd eta = (treat.array() * a1) + a0;
      double logL = (y.array() * eta - (1.0 + eta.exp()).log()).sum();

      // 3. Prior Calculations (Scalar math)
      double p_a0 = -std::log(sigma0) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow((a0 - mu0)/sigma0, 2);
      double p_a1 = -std::log(sigma1) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow((a1 - mu1)/sigma1, 2);
      double p_mu0 = -std::log(2.5) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow(mu0/2.5, 2);
      double p_mu1 = -std::log(2.5) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow(mu1/2.5, 2);
      double p_sigma0 = 0.5 * (std::log(2.0) - std::log(M_PI)) - std::log(2.5) - (sigma0*sigma0 / (2.0 * 2.5 * 2.5));
      double p_sigma1 = 0.5 * (std::log(2.0) - std::log(M_PI)) - std::log(2.5) - (sigma1*sigma1 / (2.0 * 2.5 * 2.5));

      output(i) = logL + p_a0 + p_a1 + p_mu0 + p_mu1 + p_sigma0 + p_sigma1 + jac;
    }
  }
};

// ---------------------------------------------------------------------------
//' @title Posterior Density Function using RcppParallel - Example 1
//' @name eigen_fn_log_post_1_par
//' @aliases eigen_fn_log_post_1_par
//' @description An example showing how to write a function for use with \code{\link{vegasBayesEvidence}} for
//' Bayesian computation using the RcppParallel library
//' This example function describes a simple Bayesian hierarchical model comprising of a logistic regression with
//' intercept and single binary covariate for treatment effect each with a hierarchical prior.
//' This has six parameters in total. See \code{vignette("rcpp", package = "vegasr")} for Rcpp details.
//'
//' @details The is an example function written using RcppParallel and has same functionality as the R function
//' \code{\link{fn_log_post_1}}. It uses a transformation so the density
//' can be integrated across the full domain of each parameter, i.e. the density includes a Jacobian
//'  See \code{vignette("rcpp", package = "vegasr")} for more details. Several helper function are required
//'  specifically normal and half-normal densities are also written in RcppEigen. Use Rcpp::sourceCpp()
//'  or similar to run the functions separately. They are in the fns_eigen.cpp file in the source package.
//'
//' @param theta pass a numerical R matrix of dimension Batch x M, where M is number of parameters, here M=6
//' Batch can be any positive integer
//' @param y a numeric R matrix of dimension N x 1, this is the response variable and should be 1.0 or 0.0
//' entries only
//' @param treat a numeric R matrix of dimension N x 1, this is the response variable and should be 1.0 or 0.0
//' entries only
//' @param shiftby a numerical scalar used to help avoid underflow. Used in \code{\link{vegasBayesEvidence}}
//' @param uselog a numerical flag value takes either 1.0 or 0.0 and used to return either log or real scale
//' value. Used in \code{\link{vegasBayesEvidence}}
//' @export
// Define log posterior in RcppParallel including change of variables
// [[Rcpp::export]]
Eigen::VectorXd eigen_fn_log_post_1_par(const Eigen::MatrixXd& theta,
                                         const Eigen::VectorXd& y,
                                         const Eigen::VectorXd& treat,
                                         double shiftby, double uselog){

   int n_rows = theta.rows();
   Eigen::VectorXd logPost(n_rows);

   // Parallelize the calculation across rows of theta
   LogPostWorker worker(theta, y, treat, logPost);
   RcppParallel::parallelFor(0, n_rows, worker);

   if (uselog == 1.0) {
     return (logPost.array() - shiftby).matrix();
   } else {
     return (logPost.array() - shiftby).exp().matrix();
   }
 }

// ----------------------------------------------------------------------------
// MARGINAL
// Worker for parallelizing the log-posterior calculation over the batch (theta rows)
struct LogPostWorkerM : public RcppParallel::Worker {
  const Eigen::MatrixXd& theta;
  const Eigen::VectorXd& y;
  const Eigen::VectorXd& treat;
  double z;
  Eigen::VectorXd& output;

  LogPostWorkerM(const Eigen::MatrixXd& theta, const Eigen::VectorXd& y,
                 const Eigen::VectorXd& treat, double z, Eigen::VectorXd& output)
    : theta(theta), y(y), treat(treat), z(z), output(output) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      // 1. Clamping and Transformations
      //double t0 = std::max(-0.9999, std::min(0.9999, theta(i, 0)));
      double t1 = std::max(-0.9999, std::min(0.9999, theta(i, 1-1)));
      double t2 = std::max(-0.9999, std::min(0.9999, theta(i, 2-1)));
      double t3 = std::max(-0.9999, std::min(0.9999, theta(i, 3-1)));
      double t4 = std::max(0.0001, std::min(0.9999, theta(i, 4-1)));
      double t5 = std::max(0.0001, std::min(0.9999, theta(i, 5-1)));

      // Jacobian
      double jac = //(std::log(1.0 + t0*t0) - 2.0 * std::log(1.0 - t0*t0)) +
        (std::log(1.0 + t1*t1) - 2.0 * std::log(1.0 - t1*t1)) +
        (std::log(1.0 + t2*t2) - 2.0 * std::log(1.0 - t2*t2)) +
        (std::log(1.0 + t3*t3) - 2.0 * std::log(1.0 - t3*t3)) +
        (std::log(1.0 + t4*t4) - 2.0 * std::log(1.0 - t4*t4)) +
        (std::log(1.0 + t5*t5) - 2.0 * std::log(1.0 - t5*t5));

      //double a0 = t0 / (1.0 - t0*t0);
      double a0 = z;
      double a1 = t1 / (1.0 - t1*t1);
      double mu0 = t2 / (1.0 - t2*t2);
      double mu1 = t3 / (1.0 - t3*t3);
      double sigma0 = t4 / (1.0 - t4*t4);
      double sigma1 = t5 / (1.0 - t5*t5);

      // 2. Likelihood Calculation (Vectorized for the current row)
      Eigen::ArrayXd eta = (treat.array() * a1) + a0;
      double logL = (y.array() * eta - (1.0 + eta.exp()).log()).sum();

      // 3. Prior Calculations (Scalar math)
      double p_a0 = -std::log(sigma0) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow((a0 - mu0)/sigma0, 2);
      double p_a1 = -std::log(sigma1) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow((a1 - mu1)/sigma1, 2);
      double p_mu0 = -std::log(2.5) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow(mu0/2.5, 2);
      double p_mu1 = -std::log(2.5) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow(mu1/2.5, 2);
      double p_sigma0 = 0.5 * (std::log(2.0) - std::log(M_PI)) - std::log(2.5) - (sigma0*sigma0 / (2.0 * 2.5 * 2.5));
      double p_sigma1 = 0.5 * (std::log(2.0) - std::log(M_PI)) - std::log(2.5) - (sigma1*sigma1 / (2.0 * 2.5 * 2.5));

      output(i) = logL + p_a0 + p_a1 + p_mu0 + p_mu1 + p_sigma0 + p_sigma1 + jac;
    }
  }
};

// ---------------------------------------------------------------------------
//' @title Marginal Posterior Density Function using RcppParallel - Example 1
//'
//' @description An example showing how to write a function for use with \code{\link{vegasBayesPosterior}} for
//' Bayesian computation. This is almost identical to \code{\link{eigen_fn_log_post_1_par}} but we now reduce the dimension
//' by 1 and pass a fixed value the missing dimension for the variable who marginal we want to compute.
//' See \code{vignette("rcpp", package = "vegasr")} for Rcpp details.
//' @name eigen_fn_marg_1_1_par
//' @details The is an example function written using RcppParallel and has same functionality as the R function
//' \code{\link{fn_marg_1_1}}. This function is in the fns_eigen.cpp file in the source package.
//'
//' @param theta pass a numerical R matrix of dimension Batch x M, where M is number of parameters, here M=6
//' Batch can be any positive integer
//' @param y a numeric R matrix of dimension N x 1, this is the response variable and should be 1.0 or 0.0
//' entries only
//' @param treat a numeric R matrix of dimension N x 1, this is the response variable and should be 1.0 or 0.0
//' entries only
//' @param shiftby a numerical scalar used to help avoid underflow. Used in \code{\link{vegasBayesPosterior}}
//' @param uselog a numerical flag value takes either 1.0 or 0.0 and used to return either log or real scale
//' value. Used in \code{\link{vegasBayesPosterior}}
//' @param z a numerical and the function call computes the density at this value, i.e. f(z).
//' @export
// Define log posterior in RcppParallel including change of variables
// [[Rcpp::export]]
Eigen::VectorXd eigen_fn_marg_1_1_par(const Eigen::MatrixXd& theta,
                                       const Eigen::VectorXd& y,
                                       const Eigen::VectorXd& treat,
                                       double shiftby, double uselog, double z){

   int n_rows = theta.rows();
   Eigen::VectorXd logPost(n_rows);

   // Parallelize the calculation across rows of theta
   LogPostWorkerM worker(theta, y, treat, z, logPost);
   RcppParallel::parallelFor(0, n_rows, worker);

   if (uselog == 1.0) {
     return (logPost.array() - shiftby).matrix();
   } else {
     return (logPost.array() - shiftby).exp().matrix();
   }
 }




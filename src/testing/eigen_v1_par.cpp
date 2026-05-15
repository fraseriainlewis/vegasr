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

//' Multivariate Normal Density (RcppParallel)
//'
//' @param theta   Matrix of integration variables (BATCH x N)
//' @param y  Mean vector (length N)
//' @param treat Covariance matrix (N x N)
//' @param shiftby   Scaling factor
//' @param uselog fff
//' @return vector of density values (length BATCH)
//' @export
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

//' Multivariate Normal Density (RcppParallel)
//'
//' @param theta   Matrix of integration variables (BATCH x N)
//' @param y  Mean vector (length N)
//' @param treat Covariance matrix (N x N)
//' @param shiftby   Scaling factor
//' @param uselog fff
//' @return vector of density values (length BATCH)
//' @export
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




#define EIGEN_DONT_PARALLELIZE
#include <RcppEigen.h>
#include <RcppParallel.h>
#include <cmath>

// [[Rcpp::depends(RcppEigen, RcppParallel)]]
// [[Rcpp::plugins(cpp14)]]

using Eigen::VectorXd;
using Eigen::VectorXi;
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
//' @param basket a numeric R matrix of dimension N x 1, this is the response variable and should be 1.0 or 0.0
//' entries only
//' @param shiftby a numerical scalar used to help avoid underflow. Used in \code{\link{vegasBayesEvidence}}
//' @param uselog a numerical flag value takes either 1.0 or 0.0 and used to return either log or real scale
//' value. Used in \code{\link{vegasBayesEvidence}}
//' @export
// Define log posterior in RcppEigen including change of variables
// [[Rcpp::export]]
Eigen::VectorXd eigen_fn_log_post_5(const Eigen::MatrixXd& theta,
                                     const Eigen::VectorXd& y,
                                     const Eigen::VectorXd& treat,
                                     const Eigen::VectorXd& basket,
                                     double shiftby, double uselog){

   int n_rows = theta.rows();

   // Clamp values
   //ArrayXd theta0 = theta.col(0).array().max(-0.9999).min(0.9999);
   ArrayXXd theta0 = theta.leftCols(5).array().max(-0.9999).min(0.9999); //0:4
   //ArrayXd theta1 = theta.col(1).array().max(-0.9999).min(0.9999);
   ArrayXXd theta1 = theta.middleCols(5, 5).array().max(-0.9999).min(0.9999); //5:9
   ArrayXd theta2 = theta.col(10).array().max(-0.9999).min(0.9999);
   ArrayXd theta3 = theta.col(11).array().max(-0.9999).min(0.9999);
   ArrayXd theta4 = theta.col(12).array().max(1E-05).min(0.9999);
   ArrayXd theta5 = theta.col(13).array().max(1E-05).min(0.9999);

   // Jacobian calculation using Array operations

   // ( mx.sum(mx.log1p(mx.power(theta0,2)) - 2.0*mx.log1p(-mx.power(theta0,2)),axis=0)
   //+ mx.sum(mx.log1p(mx.power(theta1,2)) - 2.0*mx.log1p(-mx.power(theta1,2)),axis=0)

   //Rcpp::Rcout <<"theta0="<<std::endl<<theta0<<std::endl;
   //Rcpp::Rcout <<"theta1="<<std::endl<<theta1<<std::endl;

   //Rcpp::Rcout <<"first="<<std::endl<< (theta0.square().log1p() - 2.0 * (-theta0.square()).log1p()).rowwise().sum()<<std::endl;

   //Rcpp::Rcout <<"second="<<std::endl<< (theta1.square().log1p() - 2.0 * (-theta1.square()).log1p()).rowwise().sum()<<std::endl;

   //Rcpp::Rcout <<"third="<<std::endl<< ((theta2.square()).log1p() - 2.0 * (-theta2.square()).log1p())<<std::endl;

   //Rcpp::Rcout <<"4="<<std::endl<< ((theta3.square()).log1p() - 2.0 * (-theta3.square()).log1p())<<std::endl;

   //Rcpp::Rcout <<"5="<<std::endl<< ((theta4.square()).log1p() - 2.0 * (-theta4.square()).log1p())<<std::endl;

   //Rcpp::Rcout <<"6="<<std::endl<< ((theta5.square()).log1p() - 2.0 * (-theta5.square()).log1p())<<std::endl;

   ArrayXd jacobianLA =  (theta0.square().log1p() - 2.0 * (-theta0.square()).log1p()).rowwise().sum().transpose()
     + (theta1.square().log1p() - 2.0 * (-theta1.square()).log1p()).rowwise().sum().transpose();

   //Rcpp::Rcout <<"JacoLA="<<jacobianLA<<std::endl;

   ArrayXd jacobianL = jacobianLA +
    // ((1.0 + theta0.square()).log() - 2.0 * (1.0 - theta0.square()).log()) +
    // ((1.0 + theta1.square()).log() - 2.0 * (1.0 - theta1.square()).log()) +
     ((theta2.square()).log1p() - 2.0 * (-theta2.square()).log1p()) +
     ((theta3.square()).log1p() - 2.0 * (-theta3.square()).log1p()) +
     ((theta4.square()).log1p() - 2.0 * (-theta4.square()).log1p()) +
     ((theta5.square()).log1p() - 2.0 * (-theta5.square()).log1p());

   //Rcpp::Rcout <<"JacoL="<<jacobianL<<std::endl;
   // Variable transformations
   MatrixXd a0 = (theta0 / (1.0 - theta0.square())).matrix();
   MatrixXd a1 = (theta1 / (1.0 - theta1.square())).matrix();

   VectorXd mu0 = (theta2 / (1.0 - theta2.square())).matrix();
   VectorXd mu1 = (theta3 / (1.0 - theta3.square())).matrix();
   VectorXd sigma0 = (theta4 / (1.0 - theta4.square())).matrix();
   VectorXd sigma1 = (theta5 / (1.0 - theta5.square())).matrix();

   // eta calculation: treat * a1.t() + a0.t()
   // treat is (N x 1), a1 is (BATCH x 1). result should be (N x BATCH)
   // We add a0.transpose() (1 x BATCH) to each row of (N x BATCH) matrix

   MatrixXd eta(treat.rows(), a0.rows());
   eta.setZero();
   // a0[0,K[0]]+a1[0,K[0]]*T[0] a0[1,K[0]]+a1[1,K[0]]*T[0] ... a0[BATCH,K[0]]+a1[BATCH,K[0]]*T[0]
   // a0[0,K[1]]+a1[0,K[1]]*T[1] a0[1,K[1]]+a1[1,K[1]]*T[0] ... a0[BATCH,K[1]]+a1[BATCH,K[1]]*T[1]

   //for (int j = 0; j < eta.cols(); ++j) {      // Iterate over columns first (better for Eigen) - batch
  //        for (int i = 0; i < eta.rows(); ++i) {  // Then rows - patients
  //              //double value = eta(i, j);           // Read
  //              int k = static_cast<int>(basket(i)) - 1;
  //              eta(i, j) = a0(j,k) + a1(j,k)*treat(i) ;            // Write
  //          }
  //    }

    // 1. Pre-calculate 0-based integer indices (Once per function call, not inside loops)
    Eigen::VectorXi k_idx = (basket.array().cast<int>() - 1);

    // 2. Iterate over the Batch (j)
    for (int j = 0; j < eta.cols(); ++j) {
          // 3. For a specific batch 'j', we need to pick intercepts/effects for all patients
          // We can use the pre-calculated k_idx to "gather" the correct parameters

          // We create temporary vectors for the current batch
          // (Size M x 1, where M is number of patients)
          Eigen::VectorXd a0_current(eta.rows());
          Eigen::VectorXd a1_current(eta.rows());

          for (int i = 0; i < eta.rows(); ++i) {
                a0_current(i) = a0(j, k_idx(i));
                a1_current(i) = a1(j, k_idx(i));
            }

          // 4. Vectorized calculation for the entire column (Patient dimension)
          // This allows Eigen to use SIMD (Single Instruction, Multiple Data)
          eta.col(j) = a0_current.array() + (a1_current.array() * treat.array());
      }


   //MatrixXd eta = (treat * a1.transpose()).rowwise() + a0.transpose();
   //MatrixXd eta = (treat * a1k.transpose()).rowwise() + a0k.transpose();

   //Rcpp::Rcout <<"eta="<<eta.rows()<<" "<<eta.cols()<<std::endl;
   //MatrixXd eta = treat.asDiagonal() * a1.col(0) + a0.col(0); //M x 5
   //Rcpp::Rcout <<"eta.size="<<eta.rows()<<" "<<eta.cols()<<std::endl;
   //Rcpp::Rcout <<"trt.size="<<treat.rows()<<std::endl;
   //Rcpp::Rcout <<"a0.size="<<a0.rows()<<" "<<a0.cols()<<std::endl;
   //Rcpp::Rcout <<"a1.size="<<a1.rows()<<" "<<a1.cols()<<std::endl;

   // term1 = y * eta (element-wise on y broadcasted across cols)
   MatrixXd term1 = eta.array().colwise() * y.array();

   // term2 = log(1 + exp(eta))
   MatrixXd term2 = (1.0 + eta.array().exp()).log();

   // logL = sum(term1 - term2, axis=0) -> returns row vector (1 x BATCH)
   Eigen::RowVectorXd logL = (term1 - term2).colwise().sum();

   // Prior calculations
   MatrixXd prior_a0 = eigen_norm_logpdf(a0, mu0, sigma0);
   MatrixXd prior_a1 = eigen_norm_logpdf(a1, mu1, sigma1);
   VectorXd prior_mu0 = eigen_norm_logpdf(mu0, 0.0, 2.5);
   VectorXd prior_mu1 = eigen_norm_logpdf(mu1, 0.0, 2.5);
   VectorXd prior_sigma0 = eigen_half_norm_logpdf(sigma0, 2.5);
   VectorXd prior_sigma1 = eigen_half_norm_logpdf(sigma1, 2.5);

   //Rcpp::Rcout <<"logL=="<<std::endl<<logL.transpose()<<std::endl;
   //Rcpp::Rcout <<"non logL"<<std::endl<< prior_a0.rowwise().sum() + prior_a1.rowwise().sum() + prior_mu0 + prior_mu1 + prior_sigma0 + prior_sigma1+ jacobianL.matrix()<<std::endl;

   //exit();
   // Final log Density
   VectorXd logDens = logL.transpose() + prior_a0.rowwise().sum() + prior_a1.rowwise().sum() + prior_mu0 + prior_mu1 + prior_sigma0 + prior_sigma1;

   // Final log Posterior
   VectorXd logPost = logDens + jacobianL.matrix();

   if (uselog == 1.0) {
     return (logPost.array() - shiftby).matrix();
   } else {
     return (logPost.array() - shiftby).exp().matrix();
   }
 }










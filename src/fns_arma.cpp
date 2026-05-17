#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]

// not exported
arma::vec arma_norm_logpdf(const arma::vec& x, const arma::vec& loc, const arma::vec& scale) {
  // 1. Calculate the squared standardized residuals: ((x - loc) / scale)^2 element-wise
  // In Armadillo, / and square() are element-wise on vectors of the same length
  arma::vec log_var = -0.5 * arma::square((x - loc) / scale);
  // 2. Calculate the log constant element-wise
  // arma::log() is vectorized
  arma::vec log_const = -arma::log(scale) - 0.5 * std::log(2.0 * M_PI);
  // 3. Return the vectorized result
  return log_var + log_const;
}

// not exported
arma::vec arma_norm_logpdf(const arma::vec& x, double loc, double scale) { //overloaded version of above
  // 1. Calculate the squared standardized residuals: ((x - loc) / scale)^2 element-wise
  // In Armadillo, / and square() are element-wise on vectors of the same length
  arma::vec log_var = -0.5 * arma::square((x - loc) / scale);
  // 2. Calculate the log constant element-wise
  // arma::log() is vectorized
  double log_const = -std::log(scale) - 0.5 * std::log(2.0 * M_PI);
  // 3. Return the vectorized result
  return log_var + log_const;
}

// not exported
arma::vec arma_half_norm_logpdf(const arma::vec& x, double sigma) {
  // 1. Calculate the log constant: log(sqrt(2/pi)) - log(sigma)
  // log(sqrt(2/pi)) = 0.5 * (log(2) - log(pi))
  double log_const = 0.5 * (std::log(2.0) - std::log(M_PI)) - std::log(sigma);

  // 2. Calculate the log exponent term: -(x^2 / (2 * sigma^2))
  // Armadillo's square(x) is vectorized
  arma::vec log_exp = - (arma::square(x) / (2.0 * sigma * sigma));

  // 3. Return result (scalar log_const is broadcasted across log_exp vector)
  return log_const + log_exp;
}


// ---------------------------------------------------------------------------
//' @title Posterior Density Function using RcppArmadillo - Example 1
//'
//' @description An example showing how to write a function for use with \code{\link{vegasBayesEvidence}} for
//' Bayesian computation using the RcppArmadillo library
//' This example function describes a simple Bayesian hierarchical model comprising of a logistic regression with
//' intercept and single binary covariate for treatment effect each with a hierarchical prior.
//' This has six parameters in total.
//'
//' @details The is an example function written using RcppArmadillo and has same functionality as the R function
//' \code{\link{fn_log_post_1}}. It uses a transformation so the density
//' can be integrated across the full domain of each parameter, i.e. the density includes a Jacobian
//'  See \code{vignette("rcpp", package = "vegasr")} for more details. Several helper function are required
//'  specifically normal and half-normal densities are also written in RcppArmadillo. Use Rcpp::sourceCpp()
//'  or similar to run the functions separately. They are in the fns_arma.cpp file in the source package.
//'
//' @title Posterior Density Function using RcppArmadillo - Example 1
//' @name arma_fn_log_post_1
//' @aliases arma_fn_log_post_1
//' @description An example showing how to write a function for use with \code{\link{vegasBayesEvidence}} for
//' Bayesian computation using the RcppArmadillo library.
//' This example function describes a simple Bayesian hierarchical model comprising of a logistic regression with
//' intercept and single binary covariate for treatment effect each with a hierarchical prior.
//' This has six parameters in total. See \code{vignette("rcpp", package = "vegasr")} for Rcpp details.
//'
//' @details The is an example function written using RcppArmadillo and has same functionality as the R function
//' \code{\link{fn_log_post_1}}. It uses a transformation so the density
//' can be integrated across the full domain of each parameter, i.e. the density includes a Jacobian
//'  See \code{vignette("rcpp", package = "vegasr")} for more details. Several helper function are required
//'  specifically normal and half-normal densities are also written in RcppArmadillo.
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
// [[Rcpp::export]]
 arma::vec arma_fn_log_post_1(const arma::mat& theta,
                              const arma::vec& y,
                              const arma::vec& treat,
                              double shiftby, double uselog) {

   //Rcpp::Rcout <<"theta batch size="<<theta.n_rows<<std::endl;

   arma::vec theta0 = arma::clamp(theta.col(0), -0.9999, 0.9999);
   arma::vec theta1 = arma::clamp(theta.col(1), -0.9999, 0.9999);
   arma::vec theta2 = arma::clamp(theta.col(2), -0.9999, 0.9999);
   arma::vec theta3 = arma::clamp(theta.col(3), -0.9999, 0.9999);
   arma::vec theta4 = arma::clamp(theta.col(4), 0.0001, 0.9999);
   arma::vec theta5 = arma::clamp(theta.col(5), 0.0001, 0.9999);
   //Rcpp::Rcout << "trimmed theta0 has " << theta0.n_elem << " rows." << std::endl;


   arma::vec jacobianL =
     (arma::log1p(arma::square(theta0)) - 2.0 * arma::log1p(-arma::square(theta0))) +
     (arma::log1p(arma::square(theta1)) - 2.0 * arma::log1p(-arma::square(theta1))) +
     (arma::log1p(arma::square(theta2)) - 2.0 * arma::log1p(-arma::square(theta2))) +
     (arma::log1p(arma::square(theta3)) - 2.0 * arma::log1p(-arma::square(theta3))) +
     (arma::log1p(arma::square(theta4)) - 2.0 * arma::log1p(-arma::square(theta4))) +
     (arma::log1p(arma::square(theta5)) - 2.0 * arma::log1p(-arma::square(theta5)));

   //Rcpp::Rcout <<"jacoL="<<jacobianL.subvec(0,3)<<std::endl;

   arma::vec a0=theta0/(1-arma::square(theta0));
   arma::vec a1=theta1/(1-arma::square(theta1));
   arma::vec mu0=theta2/(1-arma::square(theta2));
   arma::vec mu1=theta3/(1-arma::square(theta3));
   arma::vec sigma0=theta4/(1-arma::square(theta4));
   arma::vec sigma1=theta5/(1-arma::square(theta5));

   // equivalent to broadcasting from eta=a0+a1*treat; //// (10,3) where T_vec = (3,) and (10,1) broadcast to (10,3)
   arma::mat eta = treat * a1.t();
   eta.each_row() += a0.t();
   //Rcpp::Rcout <<"eta size="<<eta.n_rows<< " "<<eta.n_cols<<std::endl;

   //y_data*eta - mx.log(1+mx.exp(eta)) this is (10,3) - want col sums, so collapse over rows
   // logL = mx.sum(y*eta - mx.log1p(mx.exp(eta)),axis=0) #
   // Multiply y (200x1) element-wise against each column of eta (200x4)
   // We use .each_col() to broadcast y
   arma::mat term1 = eta;
   term1.each_col() %= y;  // %= is element-wise multiplication in Armadillo

   arma::mat term2 = arma::log1p(arma::exp(eta));
   // 3. Subtract and sum across rows (axis=0 in python is dim 0 in Armadillo)
   // sum(..., 0) returns a row vector of length 4
   arma::rowvec logL = arma::sum(term1 - term2, 0);

   /*Rcpp::Rcout <<"logL="<<logL<<std::endl;
    Rcpp::Rcout <<"a0="<<a0<<std::endl;
    Rcpp::Rcout <<"mu0="<<mu0<<std::endl;
    Rcpp::Rcout <<"sigma0="<<sigma0<<std::endl;
   */
   arma::vec prior_a0 = arma_norm_logpdf(a0,mu0,sigma0);
   arma::vec prior_a1 = arma_norm_logpdf(a1,mu1,sigma1);
   arma::vec prior_mu0 = arma_norm_logpdf(mu0,0.0,2.5);
   arma::vec prior_mu1 = arma_norm_logpdf(mu1,0.0,2.5);
   arma::vec prior_sigma0 = arma_half_norm_logpdf(sigma0, 2.5);
   arma::vec prior_sigma1 = arma_half_norm_logpdf(sigma1, 2.5);

   /*Rcpp::Rcout <<"prior_a0="<<prior_a0<<std::endl;
    Rcpp::Rcout <<"prior_a1="<<prior_a1<<std::endl;
    Rcpp::Rcout <<"prior_mu0="<<prior_mu0<<std::endl;
    Rcpp::Rcout <<"prior_mu1="<<prior_mu1<<std::endl;
    Rcpp::Rcout <<"prior_sigma0="<<prior_sigma0<<std::endl;
    Rcpp::Rcout <<"prior_sigma1="<<prior_sigma1<<std::endl;
   */
   arma::vec logDens = logL.t() + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 + prior_sigma0 + prior_sigma1;

   arma::vec logPost = logDens + jacobianL;
   //return logDens + jacobianL; //# this will be (3,)
   if(uselog==1.){ //# search phase for max - keep in log
     return(logPost - shiftby);
   } else return(arma::exp(logPost - shiftby) ); //# integrand eval - use raw


 }


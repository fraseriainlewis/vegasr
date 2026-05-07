#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp14)]]

//' Multivariate Normal Density (RcppArmadillo)
//'
//' @param x   Matrix of integration variables (BATCH x N)
//' @param mu  Mean vector (length N)
//' @param sigma Covariance matrix (N x N)
//' @return Numeric vector of density values (length BATCH)
// [[Rcpp::export]]
arma::vec dmvnorm_arma(const arma::mat& x,
                        const arma::vec& mu,
                        const arma::mat& cov,double a) {
   //const int BATCH = x.n_rows;
   const int N = x.n_cols;

   // Cholesky decomposition: sigma = U^T * U, U upper triangular
   arma::mat U = arma::chol(cov);

   // log|sigma| = 2 * sum(log(diag(U)))
   double log_det = 2.0 * arma::sum(arma::log(U.diag()));

   // Log normalising constant
   double constant = -0.5 * N * std::log(2.0 * M_PI) - 0.5 * log_det;

   // Centre observations: M x N
   arma::mat x_centered = x.each_row() - mu.t();

   // Solve U^T * z = x_centered^T  =>  z is N x M
   // Mahalanobis distance = ||z||^2 per column
   arma::mat z = arma::solve(arma::trimatl(U.t()), x_centered.t());

   arma::vec mahal = arma::sum(arma::square(z), 0).t();

   return arma::exp(constant - 0.5 * mahal)*a;
 }


#include <RcppEigen.h>
#include <cmath>

// [[Rcpp::depends(RcppEigen)]]
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

//' Multivariate Normal Density (RcppArmadillo)
//'
//' @param theta   Matrix of integration variables (BATCH x N)
//' @param y  Mean vector (length N)
//' @param treat Covariance matrix (N x N)
//' @param shiftby   Scaling factor
//' @param uselog fff
//' @return vector of density values (length BATCH)
//' @export
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

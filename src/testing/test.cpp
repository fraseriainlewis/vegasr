#include <RcppEigen.h>
#include <RcppParallel.h>
#include <cmath>

// [[Rcpp::depends(RcppEigen, RcppParallel)]]
// [[Rcpp::plugins(cpp14)]]

using namespace Eigen;

// Helper: Scalar Normal Log-PDF
inline double norm_logpdf_scalar(double x, double mu, double sigma) {
    return -std::log(sigma) - 0.5 * std::log(2.0 * M_PI) - 0.5 * std::pow((x - mu) / sigma, 2);
}

// Helper: Scalar Half-Normal Log-PDF
inline double half_norm_logpdf_scalar(double x, double sigma) {
    return 0.5 * (std::log(2.0) - std::log(M_PI)) - std::log(sigma) - (x * x / (2.0 * sigma * sigma));
}

// MARGINAL
struct LogPostWorkerM5m : public RcppParallel::Worker {
    const MatrixXd& theta;
    const VectorXd& y;
    const VectorXd& treat;
    const VectorXd& basket;
    double z;
    VectorXd& output;

    // Pre-calculated 0-indexed basket IDs
    VectorXi k_idx;

    LogPostWorkerM5m(const MatrixXd& theta, const VectorXd& y, const VectorXd& treat,
                    const VectorXd& basket, double z, VectorXd& output)
        : theta(theta), y(y), treat(treat), basket(basket), z(z), output(output) {
        // Pre-calculate indices once
        k_idx = (basket.array().cast<int>() - 1);
    }

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t ii = begin; ii < end; ++ii) {

            // 1. Extract parameters for the CURRENT batch row (ii)
            // theta row ii has 14 columns
            ArrayXd row = theta.row(ii).array();
           // ArrayXd row(row_orig.size() + 1);
            // Prepend z
            //row << z, row_orig;

            // Clamp and Transform Components
            // Intercepts (0-4) and Treatment Effects (5-9)
            ArrayXd th0 = row.segment(0, 5-1).max(-0.9999).min(0.9999);
            ArrayXd th1 = row.segment(5-1, 5).max(-0.9999).min(0.9999);
            // Hyper-parameters (10-13)
            double th2 = std::max(-0.9999, std::min(0.9999, row(10-1)));
            double th3 = std::max(-0.9999, std::min(0.9999, row(11-1)));
            double th4 = std::max(1E-05, std::min(0.9999, row(12-1)));
            double th5 = std::max(1E-05, std::min(0.9999, row(13-1)));

            // 2. Jacobian Calculation (Scalar)
            double jac = (th0.square().log1p() - 2.0 * (-th0.square()).log1p()).sum() +
                         (th1.square().log1p() - 2.0 * (-th1.array().square()).log1p()).sum() +
                         (std::log1p(th2*th2) - 2.0 * std::log1p(-th2*th2)) +
                         (std::log1p(th3*th3) - 2.0 * std::log1p(-th3*th3)) +
                         (std::log1p(th4*th4) - 2.0 * std::log1p(-th4*th4)) +
                         (std::log1p(th5*th5) - 2.0 * std::log1p(-th5*th5));

            // Variable transformations
            VectorXd a0noz_vec = (th0 / (1.0 - th0.square())).matrix();
            VectorXd a0_vec(a0noz_vec.size() + 1);
            a0_vec<< z, a0noz_vec;

            VectorXd a1_vec = (th1 / (1.0 - th1.square())).matrix();
            double mu0 = th2 / (1.0 - th2*th2);
            double mu1 = th3 / (1.0 - th3*th3);
            double sigma0 = th4 / (1.0 - th4*th4);
            double sigma1 = th5 / (1.0 - th5*th5);

            // 3. Likelihood Calculation (Patient-wise loop)
            // For this specific batch row 'ii', calculate logL over all patients
            double logL = 0.0;
            for (int i = 0; i < treat.rows(); ++i) {
                int k = k_idx(i); // Get basket index for this patient
                double eta = a0_vec(k) + a1_vec(k) * treat(i);
                logL += y(i) * eta - std::log1p(std::exp(eta));
            }

            // 4. Prior Calculations
            double prior_a0 = 0.0;
            double prior_a1 = 0.0;
            for(int k=0; k<5; ++k) {
                prior_a0 += norm_logpdf_scalar(a0_vec(k), mu0, sigma0);
                prior_a1 += norm_logpdf_scalar(a1_vec(k), mu1, sigma1);
            }

            double prior_hyper = norm_logpdf_scalar(mu0, 0.0, 2.5) +
                                 norm_logpdf_scalar(mu1, 0.0, 2.5) +
                                 half_norm_logpdf_scalar(sigma0, 2.5) +
                                 half_norm_logpdf_scalar(sigma1, 2.5);

            // Final log Posterior for this batch entry
            output(ii) = logL + prior_a0 + prior_a1 + prior_hyper + jac;
        }
    }
};

//' @export
// [[Rcpp::export]]
Eigen::VectorXd eigen_fn_log_post_5m_par(const Eigen::MatrixXd& theta,
                                         const Eigen::VectorXd& y,
                                         const Eigen::VectorXd& treat,
                                         const Eigen::VectorXd& basket,
                                         double shiftby, double uselog, double z){

   int n_rows = theta.rows();
   Eigen::VectorXd logPost(n_rows);

   LogPostWorkerM5m worker(theta, y, treat, basket, z, logPost);
   RcppParallel::parallelFor(0, n_rows, worker);

   if (uselog == 1.0) {
     return (logPost.array() - shiftby).matrix();
   } else {
     return (logPost.array() - shiftby).exp().matrix();
   }
 }

// -------------------------------------------





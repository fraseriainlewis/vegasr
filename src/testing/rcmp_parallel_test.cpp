// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <cmath>

using namespace RcppParallel;

struct ParallelExp : public Worker {
   // Input and Output (Pointers to Armadillo data)
   const arma::vec& input;
   arma::vec& output;

   // Constructor
   ParallelExp(const arma::vec& input, arma::vec& output) 
      : input(input), output(output) {}

   // Parallel function operator
   void operator()(std::size_t begin, std::size_t end) {
      // Clang will auto-vectorize this loop using SIMD (NEON)
      for (std::size_t i = begin; i < end; i++) {
         output[i] = std::exp(input[i]);
      }
   }
};

// [[Rcpp::export]]
arma::vec parallel_arma_exp(const arma::vec& x) {
   int n = x.n_elem;
   arma::vec res(n);

   // Create the worker
   ParallelExp worker(x, res);
   
   // Execute in parallel
   // 250,000 is large enough that 'parallelFor' will 
   // split this into optimal chunks for your CPU cores.
   parallelFor(0, n, worker);
   
   return res;
}

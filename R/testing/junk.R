
fn_log_post_K <- function(theta, y, treat, basketID, shiftby, uselog, K= NULL) {

  y <- as.numeric(y)
  treat <- as.numeric(treat)
  basketID <- as.integer(basketID)


  if (is.null(K)) K <- max(basketID, na.rm = TRUE)
  if (any(basketID < 1L | basketID > K)) stop("basketID must be in 1..K")

  B <- nrow(theta)
  M_expected <- 2*K + 4
  if (ncol(theta) != M_expected) {
    stop(sprintf("theta must have %d columns (2*K+4). Got %d.", M_expected, ncol(theta)))
  }

  # split theta
  theta0 <- pmin(pmax(theta[, 1:K, drop=FALSE],           -0.9999), 0.9999)
  theta1 <- pmin(pmax(theta[, (K+1):(2*K), drop=FALSE],   -0.9999), 0.9999)
  theta2 <- pmin(pmax(theta[, 2*K + 1],                  -0.9999), 0.9999)
  theta3 <- pmin(pmax(theta[, 2*K + 2],                  -0.9999), 0.9999)
  theta4 <- pmin(pmax(theta[, 2*K + 3],                   1e-5),   0.9999)
  theta5 <- pmin(pmax(theta[, 2*K + 4],                   1e-5),   0.9999)

  log_jac_term <- function(th) log1p(th^2) - 2*log1p(-(th^2))


  jacobianLB <- rowSums(log_jac_term(theta0)) +
    rowSums(log_jac_term(theta1)) +
    log_jac_term(theta2) + log_jac_term(theta3) + log_jac_term(theta4) + log_jac_term(theta5)

  #cat("jac\n")
  #print(rowSums(log_jac_term(theta0)) +
  #        rowSums(log_jac_term(theta1)))

  #print(jacobianLB)


  a0 <- theta0/(1 - theta0^2)   # B x K
  a1 <- theta1/(1 - theta1^2)   # B x K
  mu0 <- theta2/(1 - theta2^2)  # length B
  mu1 <- theta3/(1 - theta3^2)
  sigma0 <- theta4/(1 - theta4^2)
  sigma1 <- theta5/(1 - theta5^2)

  treat_BN <- matrix(rep(treat, each = B), nrow = B)  # B x N
  eta_BN <- a0[, basketID, drop=FALSE] + a1[, basketID, drop=FALSE] * treat_BN
  eta <- t(eta_BN)  # N x B
  #cat("eta=\n")
  #print(dim(eta))

  log1pexp <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
  logL <- colSums(y * eta - log1p(exp(eta)))  # length B

  # priors: dnorm will recycle mu0/sigma0 across columns (K)
  prior_a0 <- stats::dnorm(a0, mean = mu0, sd = sigma0, log = TRUE)
  prior_a1 <- rowSums(stats::dnorm(a1, mean = mu1, sd = sigma1, log = TRUE))
  prior_mu0 <- stats::dnorm(mu0, mean = 0, sd = 2.5, log = TRUE)
  prior_mu1 <- stats::dnorm(mu1, mean = 0, sd = 2.5, log = TRUE)
  prior_sigma0 <- extraDistr::dhnorm(sigma0, sigma = 2.5, log = TRUE)
  prior_sigma1 <- extraDistr::dhnorm(sigma1, sigma = 2.5, log = TRUE)

  #cat("logL\n")
  #print(logL)
  #cat("hhh\n")
  #print(a0)
  #cat("prior tot+jac\n")
  #print(rowSums(prior_a0) + prior_a1+prior_mu0 + prior_mu1 + prior_sigma0 + prior_sigma1+ jacobianLB)

  logPost <- logL + rowSums(prior_a0) + prior_a1 + prior_mu0 + prior_mu1 + prior_sigma0 + prior_sigma1 + jacobianLB

  if (uselog == 1) return(logPost - shiftby[1])
  exp(logPost - shiftby[1])
}










#Function for the log posterior for K baskets
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

  a0 <- theta0/(1 - theta0^2)   # B x K
  a1 <- theta1/(1 - theta1^2)   # B x K
  mu0 <- theta2/(1 - theta2^2)  # length B
  mu1 <- theta3/(1 - theta3^2)
  sigma0 <- theta4/(1 - theta4^2)
  sigma1 <- theta5/(1 - theta5^2)

  treat_BN <- matrix(rep(treat, each = B), nrow = B)  # B x N
  eta_BN <- a0[, basketID, drop=FALSE] + a1[, basketID, drop=FALSE] * treat_BN
  eta <- t(eta_BN)  # N x B

  log1pexp <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
  logL <- colSums(y * eta - log1p(exp(eta)))  # length B

  # priors: dnorm will recycle mu0/sigma0 across columns (K)
  prior_a0 <- rowSums(stats::dnorm(a0, mean = mu0, sd = sigma0, log = TRUE))
  prior_a1 <- rowSums(stats::dnorm(a1, mean = mu1, sd = sigma1, log = TRUE))
  prior_mu0 <- stats::dnorm(mu0, mean = 0, sd = 2.5, log = TRUE)
  prior_mu1 <- stats::dnorm(mu1, mean = 0, sd = 2.5, log = TRUE)
  prior_sigma0 <- extraDistr::dhnorm(sigma0, sigma = 2.5, log = TRUE)
  prior_sigma1 <- extraDistr::dhnorm(sigma1, sigma = 2.5, log = TRUE)

  logPost <- logL + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 + prior_sigma0 + prior_sigma1 + jacobianLB

  if (uselog == 1) return(logPost - shiftby[1])
  exp(logPost - shiftby[1])
}

#Function for finding the  beta[2,2] marginal density

fn_marg_a1_2 <- function(theta, y, treat, basketID, shiftby, uselog, z, K = NULL) {

  y <- as.numeric(y)
  treat <- as.numeric(treat)
  basketID <- as.integer(basketID)


  if (is.null(K)) K <- max(basketID, na.rm = TRUE)
  if (K < 2) stop("Need K >= 2 to marginalize a1[,2].")
  if (any(basketID < 1L | basketID > K)) stop("basketID must be in 1..K")

  B <- nrow(theta)

  # We removed theta1[,2], so dimension is (2*K + 3):
  # theta0: K cols
  # theta1_free: (K-1) cols (all baskets except 2)
  # theta2..theta5: 4 scalars
  M_expected <- 2*K + 3
  if (ncol(theta) != M_expected) {
    stop(sprintf("theta must have %d columns (2*K+3). Got %d.", M_expected, ncol(theta)))
  }

  # split theta
  theta0 <- pmin(pmax(theta[, 1:K, drop=FALSE], -0.9999), 0.9999)

  # theta1 for baskets except 2, in order: 1,3,4,...,K
  k_free <- setdiff(seq_len(K), 2L)
  theta1_free <- pmin(pmax(theta[, (K+1):(K+(K-1)), drop=FALSE], -0.9999), 0.9999)

  # remaining scalars (shifted by -1 compared to full model)
  theta2 <- pmin(pmax(theta[, 2*K],     -0.9999), 0.9999)  # mu0
  theta3 <- pmin(pmax(theta[, 2*K + 1], -0.9999), 0.9999)  # mu1
  theta4 <- pmin(pmax(theta[, 2*K + 2],  1e-5),   0.9999)  # sigma0
  theta5 <- pmin(pmax(theta[, 2*K + 3],  1e-5),   0.9999)  # sigma1

  log_jac_term <- function(th) log1p(th^2) - 2*log1p(-(th^2))

  # Jacobian: include theta0, theta1_free, theta2..theta5; exclude removed theta1[,2]
  jacobian <- rowSums(log_jac_term(theta0)) +
    rowSums(log_jac_term(theta1_free)) +
    log_jac_term(theta2) + log_jac_term(theta3) + log_jac_term(theta4) + log_jac_term(theta5)

  a0 <- theta0/(1 - theta0^2)  # B x K

  a1 <- matrix(NA_real_, nrow = B, ncol = K)
  a1[, 2] <- z
  a1[, k_free] <- theta1_free/(1 - theta1_free^2)  # fill other baskets

  mu0 <- theta2/(1 - theta2^2)
  mu1 <- theta3/(1 - theta3^2)
  sigma0 <- theta4/(1 - theta4^2)
  sigma1 <- theta5/(1 - theta5^2)

  treat_BN <- matrix(rep(treat, each = B), nrow = B)  # B x N
  eta_BN <- a0[, basketID, drop=FALSE] + a1[, basketID, drop=FALSE] * treat_BN
  eta <- t(eta_BN)  # N x B

  logL <- colSums(y * eta - log1p(exp(eta)))

  prior_a0 <- rowSums(stats::dnorm(a0, mean = mu0, sd = sigma0, log = TRUE))
  prior_a1 <- rowSums(stats::dnorm(a1, mean = mu1, sd = sigma1, log = TRUE))
  prior_mu0 <- stats::dnorm(mu0, mean = 0, sd = 2.5, log = TRUE)
  prior_mu1 <- stats::dnorm(mu1, mean = 0, sd = 2.5, log = TRUE)
  prior_sigma0 <- extraDistr::dhnorm(sigma0, sigma = 2.5, log = TRUE)
  prior_sigma1 <- extraDistr::dhnorm(sigma1, sigma = 2.5, log = TRUE)

  logPost <- logL + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 + prior_sigma0 + prior_sigma1 + jacobian

  if (uselog == 1) return(logPost - shiftby[1])
  exp(logPost - shiftby[1])
}




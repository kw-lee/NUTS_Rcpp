// #define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]

using namespace Rcpp;
using namespace RcppArmadillo;

// [[Rcpp::export]]
arma::vec grad_f(arma::vec& theta) {
    return  1.0 - arma::exp(2 * theta);
}

// [[Rcpp::export]]
double f(arma::vec& theta) {
    // - log_normal
    return arma::sum(theta) - arma::sum(0.5 * arma::exp(2 * theta));
}

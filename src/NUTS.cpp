// #define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
#include <progress.hpp>
#include <math.h>
#include <tuple>
#include <iostream>
#include <progress_bar.hpp>
#include "function.h"

#if !defined(WIN32) && !defined(__WIN32) && !defined(__WIN32__)
#include <Rinterface.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]

using namespace Rcpp;
using namespace RcppArmadillo;

class ETAProgressBar: public ProgressBar{

  public:
    /**
    * Main constructor
    */
    ETAProgressBar()  { reset(); }

    ~ETAProgressBar() {}

  public: // ===== main methods =====
    
    void display() {
        REprintf("0%%   10   20   30   40   50   60   70   80   90   100%%\n");
        REprintf("[----|----|----|----|----|----|----|----|----|----|\n");
        flush_console();
    }

    void reset() {
        _max_ticks = 50;
        _finalized = false;
        _timer_flag = true;
    }

    void update(float progress) {
    
        // stop if already finalized
        if (_finalized) return;
    
        // start time measurement when update() is called the first time
        if (_timer_flag) {
            _timer_flag = false;
            // measure start time
            time(&start);
        } else {
        
            // measure current time
            time(&end);
        
            // calculate passed time and remaining time (in seconds)
            double pas_time = std::difftime(end, start);
            double rem_time = (pas_time / progress) * (1 - progress);
        
            // convert seconds to time string
            std::string pas_time_string = _time_to_string(pas_time);
            std::string time_string = _time_to_string(rem_time);
        
            // create progress bar string 
            std::string progress_bar_string = _current_ticks_display(progress);
        
            // ensure overwriting of old time info
            int empty_length = time_string.length();
            std::string empty_space = std::string(empty_length, ' ');
        
            // merge progress bar and time string
            std::stringstream strs;
            strs << "|" << progress_bar_string << "| Elapsed: " << pas_time_string << ", Remaining: " << time_string << empty_space;
            std::string temp_str = strs.str();
            char const* char_type = temp_str.c_str();
        
            // print: remove old display and replace with new
            REprintf("\r");
            REprintf("%s", char_type);
        
            // finalize display when ready
            if(progress == 1) {
                _finalize_display();
            }  
        }
    }

    void _finalize_display() {
        if (_finalized) return;

        REprintf("|\n");
        flush_console();
        _finalized = true;
    }

    std::string _time_to_string(double seconds) {
        int time = (int) seconds;
    
        int hour = 0;
        int min = 0;
        int sec = 0;
    
        hour = time / 3600;
        time = time % 3600;
        min = time / 60;
        time = time % 60;
        sec = time;
    
        std::stringstream time_strs;
        if (hour != 0) time_strs << hour << "h ";
        if (min != 0) time_strs << min << "min ";
        if (sec != 0) time_strs << sec << "s ";
        std::string time_str = time_strs.str();
    
        return time_str;
    }

    std::string _current_ticks_display(float progress) {
        int nb_ticks = _compute_nb_ticks(progress);
        std::string cur_display = _construct_ticks_display_string(nb_ticks);
        return cur_display;
    }

    int _compute_nb_ticks(float progress) {
        return int(progress * _max_ticks);
    }

    std::string _construct_ticks_display_string(int nb) {
        std::stringstream ticks_strs;
        for (int i = 0; i < (_max_ticks - 1); ++i) {
            if (i < nb) {
                ticks_strs << "*";
            } else {
                ticks_strs << " ";
            }
        }
        std::string tick_space_string = ticks_strs.str();
        return tick_space_string;
    }

    void flush_console() {
        #if !defined(WIN32) && !defined(__WIN32) && !defined(__WIN32__)
                R_FlushConsole();
        #endif
    }

    void end_display() {
        if (_finalized) return;
        REprintf("\n");
        _finalized = true;
    }
  
  private: 

    int _max_ticks;
    bool _finalized;
    bool _timer_flag;
    time_t start, end;
       
};

arma::field<arma::vec> leapfrog_step(arma::vec& theta, arma::vec& r, double eps, arma::vec& M_diag) {
    arma::field<arma::vec> param(2);
    arma::vec r_tilde = r + 0.5 * eps * grad_f(theta);
    arma::vec theta_tilde = theta + eps * (r_tilde / M_diag);
    r_tilde = r_tilde + 0.5 * eps * grad_f(theta_tilde);
    param(0) = theta_tilde; 
    param(1) = r_tilde;
    return param; 
}

double joint_log_density (arma::vec& theta, arma::vec& r, arma::vec& M_diag) {
    return f(theta) - 0.5 * arma::dot(r, r / M_diag);
}

double find_reasonable_epsilon(arma::vec& theta, arma::vec& M_diag, double eps0 = 1.0, bool verbose=true) {
    double eps = eps0;
    arma::vec r = arma::randn(theta.n_elem) % arma::sqrt(M_diag);
    arma::field<arma::vec> proposed = leapfrog_step(theta, r, eps, M_diag);
    double log_ratio = joint_log_density(proposed(0), proposed(1), M_diag);
    double alpha = (std::exp(log_ratio) > 0.5) ? 1.0 : -1.0;
    int count = 1;
    while (alpha * log_ratio > (-alpha) * std::log(2.0)) {
        eps = std::pow(2.0, alpha) * eps;
        proposed = leapfrog_step(theta, r, eps, M_diag);
        log_ratio = joint_log_density(proposed(0), proposed(1), M_diag) - joint_log_density(theta, r, M_diag);
        if (count > 100) {
            Rcpp::Rcout << "Could not find reasonable epsilon in 100 iterations!" << std::endl;
            break;
        }
        count++;
    }
    if (verbose) {
        Rcpp::Rcout << "reasonable epsilon = " << eps << " found after " << count << " steps" << std::endl;
    }
    return eps;
}

bool check_NUTS(bool s, arma::vec& theta_plus, arma::vec& theta_minus, arma::vec& r_plus, arma::vec& r_minus) {
    bool condition1 = (arma::dot(theta_plus - theta_minus, r_minus) >= 0);
    bool condition2 = (arma::dot(theta_plus - theta_minus, r_plus) >= 0);
    return s && condition1 && condition2;
}

std::tuple<arma::vec, arma::vec, arma::vec, arma::vec, arma::vec, bool, int, double, int> build_tree(
    arma::vec theta, arma::vec r, double u, int v, int j, double eps0, arma::vec theta0, arma::vec r0, arma::vec M_diag, double DELTA_MAX=1000.0
) {
    if (j == 0) {
        arma::field<arma::vec> proposed = leapfrog_step(theta, r, v * eps0, M_diag);
        arma::vec theta = proposed(0);
        arma::vec r = proposed(1);
        double log_prob = joint_log_density(theta, r, M_diag);
        double log_prob0 = joint_log_density(theta0, r0, M_diag);
        bool n = (std::log(u) <= log_prob);
        bool s = (std::log(u) < DELTA_MAX + log_prob);
        double alpha = std::min(1.0, std::exp(log_prob - log_prob0));
        return std::make_tuple(theta, theta, theta, r, r, s, n, alpha, 1);
    } else {
        arma::vec theta_minus; 
        arma::vec theta_plus;
        arma::vec r_minus; 
        arma::vec r_plus; 
        bool s;
        int n;
        double alpha; 
        int n_alpha;
        arma::vec theta_minus1; 
        arma::vec theta_plus1;
        arma::vec theta1; 
        arma::vec r_minus1; 
        arma::vec r_plus1; 
        bool s1;
        int n1;
        double alpha1; 
        int n_alpha1;
        std::tie(theta_minus, theta_plus, theta, r_minus, r_plus, s, n, alpha, n_alpha) = build_tree(theta, r, u, v, j-1, eps0, theta0, r0, M_diag);
        if (s) {
            if (v == -1) {
                std::tie(theta_minus, theta_plus1, theta1, r_minus, r_plus1, s1, n1, alpha1, n_alpha1) = build_tree(theta_minus, r_minus, u, v, j-1, eps0, theta0, r0, M_diag);
            } else {
                std::tie(theta_minus1, theta_plus, theta1, r_minus1, r_plus, s1, n1, alpha1, n_alpha1) = build_tree(theta_plus, r_plus, u, v, j-1, eps0, theta0, r0, M_diag);
            }
            n = n + n1;
            if (n != 0) {
                double prob = (double) (n1 / n);
                if ((((double) rand() / (RAND_MAX))) < prob) {
                    theta = theta1;
                }
            }
            s = check_NUTS(s1, theta_plus, theta_minus, r_plus, r_minus);
            alpha = alpha + alpha1;
            n_alpha = n_alpha + n_alpha1;
        } 

        return std::make_tuple(theta_minus, theta_plus, theta, r_minus, r_plus, s, n, alpha, n_alpha);
    }
}

std::tuple<arma::vec, double, double, double, double, int, arma::vec> NUTS_one_step(
    arma::vec theta, int iter, double eps1, double eps_bar1, double H1, double mu1, int M_adapt, arma::vec M_diag, double delta = 0.5, int max_treedepth = 10, double eps0 = 1.0, bool verbose = true
) {
    int n_dim = theta.n_elem;
    double kappa = 0.75;
    double t0 = 10.0;
    double gamma = 0.05;
    double eps;
    double mu;
    double H;
    double eps_bar;
    
    if (M_diag.is_empty()) {
        M_diag = arma::ones(n_dim);
    } 

    if (iter == 1) {
        eps = find_reasonable_epsilon(theta, M_diag, eps0, verbose);
        mu = std::log(10.0 * eps);
        H = 0.0;
        eps_bar = 1.0;
    } else {
        eps = eps1;
        eps_bar = eps_bar1;
        H = H1;
        mu = mu1;
    }

    arma::vec r0 = arma::randn(n_dim) % arma::sqrt(M_diag);
    double u = std::exp(f(theta) - 0.5 * arma::dot(r0, r0 / M_diag)) * (((double) rand() / (RAND_MAX)));
    // if (!std::isnormal(u)) {
    //     Rcout << "NUTS: sampled slice u is NaN!" << std::endl;
    //     u = 1e5 * (((double) rand() / (RAND_MAX)) + 1);
    // }

    arma::vec theta_minus = theta;
    arma::vec theta_plus = theta;
    arma::vec r_minus = r0;
    arma::vec r_plus = r0;

    arma::vec theta_minus1 = theta;
    arma::vec theta_plus1 = theta;
    arma::vec theta1 = theta;
    arma::vec r_minus1 = r0;
    arma::vec r_plus1 = r0;

    if (iter > M_adapt) {
        eps = 0.9 * eps_bar + 0.2 * (((double) rand() / (RAND_MAX)));
    }


    int j = 0;
    int n = 1;
    bool s = true;
    int direction;
    bool s1; 
    int n1;
    double alpha1;
    int n_alpha1;
    double log_eps;

    while (s) {
        direction = (std::rand() < (RAND_MAX / 2)) ? -1 : 1;
        if (direction == -1) {
            std::tie(theta_minus, theta_plus1, theta1, r_minus, r_plus1, s1, n1, alpha1, n_alpha1) = build_tree(theta_minus, r_minus, u, direction, j, eps, theta, r0, M_diag);
        } else {
            std::tie(theta_minus1, theta_plus, theta1, r_minus1, r_plus, s1, n1, alpha1, n_alpha1) = build_tree(theta_minus, r_minus, u, direction, j, eps, theta, r0, M_diag);
        }

        if (s1) {
            if ((((double) rand() / (RAND_MAX))) < n1 / n) {
                theta = theta1;
            }
        }
        n = n + n1;
        s = check_NUTS(s1, theta_plus, theta_minus, r_plus, r_minus);
        j++;

        if (j > max_treedepth) {
            // Rcout << "NUTS: Reached max tree depth!" << std::endl;
            break;
        }
    }

    if (iter <= M_adapt) {
        H = (1.0 - 1.0 / ((double) iter + t0)) * H + (delta - alpha1 / n_alpha1 ) / ((double) iter + t0);
        log_eps = mu - std::sqrt((double) iter) / gamma * H;
        eps_bar = std::exp(std::pow(iter, -kappa) * log_eps + (1 - std::pow(iter, -kappa)) * std::log(eps_bar));
        eps = std::exp(log_eps);
    } else {
        eps = eps_bar;
    }

    return std::make_tuple(theta, eps, eps_bar, H, mu, M_adapt, M_diag);

}

// [[Rcpp::export]]
arma::mat NUTS(arma::vec theta, int n_iter, arma::vec M_diag0, int M_adapt = 50, double delta = 0.5, int max_treedepth = 10, double eps0 = 1.0, bool verbose = true, bool display_progress = true) {
    arma::mat theta_trace(theta.n_elem, n_iter, arma::fill::zeros);
    ETAProgressBar pb;
    Progress p(n_iter, display_progress, pb);
    double eps_bar = 0.0; 
    double H = 0.0;
    double mu = 0.0; 
    double eps = eps0;
    arma::vec M_diag = M_diag0;
    for (int iter = 0; iter < n_iter; iter++) {
        std::tie(theta, eps, eps_bar, H, mu, M_adapt, M_diag) = NUTS_one_step(theta, iter, eps, eps_bar, H, mu, M_adapt, M_diag, delta, max_treedepth, eps, verbose);
        theta_trace.col(iter) = theta;
        p.increment(); 
    }
    return theta_trace;
}
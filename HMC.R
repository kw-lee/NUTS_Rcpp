library(dplyr)
library(tictoc)
library(ggplot2)
library(gridExtra)

Rcpp::sourceCpp("src/function.cpp")
Rcpp::sourceCpp("src/NUTS.cpp")

theta0 <- c(0, 0)
n_iter <- as.integer(1e5)
M_adapt <- as.integer(1e4)
M_diag0 <- c(1, 1)
max_treedepth <- 10

tic()
theta <- NUTS(theta0, n_iter, M_diag0, M_adapt, max_treedepth)
toc()

theta <- theta %>% t() %>% as.data.frame()
colnames(theta) <- c("theta1", "theta2")
theta$idx <- 1:n_iter

# theta <- theta %>% filter(idx > M_adapt)

g1 <- ggplot(data = theta) +
    geom_line(aes(x = idx, y = theta1))
g2 <- ggplot(data = theta) +
    geom_line(aes(x = idx, y = theta2))
grid.arrange(g1, g2)

theta %>%
    mutate(theta1 = exp(theta1), theta2 = exp(theta2)) %>%
    ggplot(mapping = aes(x = theta1, y = theta2)) +
    geom_point()

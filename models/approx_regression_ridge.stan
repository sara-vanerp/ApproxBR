data {
  int p; // number of parameters
  vector[p] mle; // mle estimates
  matrix[p, p] errorcov; // error covariance matrix
  real<lower = 0> s0; // scale for the half-t on the prior SD
  int<lower = 0> nu0; // degrees of freedom for the half-t on the prior SD
}

parameters {
  vector[p] theta;
  real<lower = 0> sd_ridge;
}

model {
  target += student_t_lpdf(sd_ridge | nu0, 0, s0);
  target += normal_lpdf(theta | 0, sd_ridge);
  target += multi_normal_lpdf(theta | mle, errorcov);
}


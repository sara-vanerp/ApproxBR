data {
  int p; // number of parameters
  vector[p] mle; // mle estimates
  matrix[p, p] errorcov; // error covariance matrix
  real<lower = 0> s0; // scale for the half-t on the penalty parameter
  int<lower = 0> nu0; // degrees of freedom for the half-t on the penalty parameter
}

parameters {
  vector[p] theta;
  real<lower = 0> lambda;
}

model {
  target += student_t_lpdf(lambda | nu0, 0, s0);
  target += double_exponential_lpdf(theta | 0, 1/lambda);
  target += multi_normal_lpdf(theta | mle, errorcov);
}


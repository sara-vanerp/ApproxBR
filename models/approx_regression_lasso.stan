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

transformed parameters {
  vector[p] scale_lasso;
  for(i in 1:p){
    scale_lasso[i] = errorcov[i, i]/lambda; // 
  }
}

model {
  target += student_t_lpdf(lambda | nu0, 0, s0);
  target += double_exponential_lpdf(theta | 0, scale_lasso);
  target += multi_normal_lpdf(theta | mle, errorcov);
}


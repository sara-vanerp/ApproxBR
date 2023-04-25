data {
  int p; // number of parameters
  vector[p] mle; // mle estimates
  matrix[p, p] errorcov; // error covariance matrix
  real<lower = 0> s0; // scale for the cauchy on lambda
}

parameters {
  vector[p] theta_raw;
  //hyperparameters prior
  vector<lower=0, upper=pi()/2>[p] tau_unif;
  real<lower=0> lambda;
}

transformed parameters{
  vector[p] theta;
  vector<lower=0>[p] tau;
  tau = lambda * tan(tau_unif); // implies tau ~ cauchy(0, lambda)
  for(j in 1:p){
    theta[j] = tau[j] * theta_raw[j];
  }
}

model {
  target += cauchy_lpdf(lambda | 0, s0);
  target += normal_lpdf(theta_raw | 0, 1); // implies theta ~ normal(0, tau)
  target += multi_normal_lpdf(theta | mle, errorcov);
}


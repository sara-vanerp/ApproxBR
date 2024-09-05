data {
  int<lower = 1> N; 
  int<lower = 1> nM;
  vector[N] X;
  matrix[N, nM] M;
  vector[N] Y;
  real<lower = 0> s0; // scale for the cauchy on lambda
}
parameters {
  vector[nM] a; 
  vector<lower = 0>[nM] sigM;
  vector[nM] b;
  real c; //direct effect
  real<lower = 0> sigY;
  //hyperparameters prior
  vector<lower=0, upper=pi()/2>[nM] tau_unif;
  real<lower=0> lambda;
}
transformed parameters {
  matrix[N, nM] meanM;
  vector[N] meanY;
  vector[nM] ab_raw;
  vector<lower=0>[nM] tau;
  vector[nM] ab;
  tau = lambda * tan(tau_unif); // implies tau ~ cauchy(0, lambda)
  for(m in 1:nM){
    meanM[, m] = a[m] * X;
    ab_raw[m] = a[m]*b[m];
    ab[m] = tau[m] * ab_raw[m];
  }
  meanY = M*b + c*X;
}
model {
  // prior on the indirect effects
  target += cauchy_lpdf(lambda | 0, s0);
  target += normal_lpdf(ab_raw | 0, 1); // implies theta ~ normal(0, tau)
  // model for M
  target += student_t_lpdf(sigM | 3, 0, 10);
  for(m in 1:nM){
    target += normal_lpdf(M[, m] | meanM[, m], sigM[m]);
  }
  // model for Y
  target += normal_lpdf(c | 0, 100); // direct effect
  target += student_t_lpdf(sigY | 3, 0, 10);
  target += normal_lpdf(Y | meanY, sigY);
}

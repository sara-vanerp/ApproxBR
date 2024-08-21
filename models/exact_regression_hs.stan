data{
	int N_train; //number of observations training and validation set
	int p; //number of predictors
	real y_train[N_train]; //response vector
	matrix[N_train, p] X_train; //model matrix
  	real<lower = 0> s0; // scale for the cauchy on lambda
}
parameters{
	real<lower = 0> sigma2; //error variance
	vector[p] beta_raw; // regression parameters
	//hyperparameters prior
	vector<lower=0, upper=pi()/2>[p] tau_unif;
	real<lower=0> lambda;
}
transformed parameters{
	vector[p] beta;
	real<lower = 0> sigma; //error sd
	vector<lower=0>[p] tau;
	vector[N_train] linpred; //mean normal model
	tau = lambda * tan(tau_unif); // implies tau ~ cauchy(0, lambda)
	for(j in 1:p){
		beta[j] = tau[j] * beta_raw[j];
	}
	sigma = sqrt(sigma2);
	linpred = X_train*beta;
}
model{
 //prior regression coefficients: horseshoe
  target += cauchy_lpdf(lambda | 0, s0);
  target += normal_lpdf(beta_raw | 0, 1);
	
 //priors nuisance parameters: uniform on log(sigma^2) 
	target += -2 * log(sigma); 
	
 //likelihood
	y_train ~ normal(linpred, sigma);
}

  

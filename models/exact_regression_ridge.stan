data{
	int N_train; //number of observations training and validation set
	int p; //number of predictors
	real y_train[N_train]; //response vector
	matrix[N_train, p] X_train; //model matrix
  	real<lower = 0> s0; // scale for the half-t on the prior SD
    int<lower = 0> nu0; // degrees of freedom for the half-t on the prior SD
}
parameters{
	real<lower = 0> sigma2; //error variance
	vector[p] beta; // regression parameters
	//hyperparameters prior
	real<lower = 0> sd_ridge; //penalty parameter
}
transformed parameters{
	real<lower = 0> sigma; //error sd
	vector[N_train] linpred; //mean normal model
	sigma = sqrt(sigma2);
	linpred = X_train*beta;
}
model{
 //prior regression coefficients: ridge
  target += student_t_lpdf(sd_ridge | nu0, 0, s0);
  target += normal_lpdf(beta | 0, sd_ridge);
	
 //priors nuisance parameters: uniform on log(sigma^2) 
	target += -2 * log(sigma); 
	
 //likelihood
	y_train ~ normal(linpred, sigma);
}

  

##### Replicating the Bayesian regularization regression simulation to evaluate shrinkem

library(utils)
library(parallel)
library(rstan)
rstan_options(auto_write = TRUE) # to avoid recompiling stan model
library(dplyr)

##### Generate data
# condition 1 of Van Erp et al. (2019)
set.seed(123)
beta_true <- c(3,1.5,0,0,2,0,0,0)
k <- length(beta_true)
sigma_true <- 3
n_train <- 40
n_test <- 200
n <- n_train+n_test
reps <- 500
df1 <- vector(mode='list', length=reps)

for(s in 1:reps){
  
  # generate predictors
  X_obs <- matrix(rnorm(n*k),ncol=k)
  X_train <- X_obs[1:n_train,]
  X_test <- X_obs[n_train+1:n_test,]
  
  # generate outcomes
  y_train <- c(X_train %*% beta_true) + rnorm(n_train,sd=sigma_true)
  y_test <- c(X_test %*% beta_true) + rnorm(n_test,sd=sigma_true)
  
  df1[[s]] <- list(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  
}

save(df1, file = "./analyses_R1/df1.RData")

# condition 1 of Van Erp et al. (2019), but with smaller n_train
set.seed(123)
beta_true <- c(3,1.5,0,0,2,0,0,0)
k <- length(beta_true)
sigma_true <- 3
n_train <- 10
n_test <- 200
n <- n_train+n_test
reps <- 500
df2 <- vector(mode='list', length=reps)

for(s in 1:reps){
  
  # generate predictors
  X_obs <- matrix(rnorm(n*k),ncol=k)
  X_train <- X_obs[1:n_train,]
  X_test <- X_obs[n_train+1:n_test,]
  
  # generate outcomes
  y_train <- c(X_train %*% beta_true) + rnorm(n_train,sd=sigma_true)
  y_test <- c(X_test %*% beta_true) + rnorm(n_test,sd=sigma_true)
  
  df2[[s]] <- list(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  
}

save(df2, file = "./analyses_R1/df2.RData")

# condition 3 of Van Erp et al. (2019)
set.seed(123)
beta_true <- c(rep(3,15),rep(0,15))
k <- length(beta_true)
sigma_true <- sqrt(225)
n_train <- 200
n_test <- 400
n <- n_train+n_test
reps <- 500
df3 <- vector(mode='list', length=reps)

for(s in 1:reps){
  
  # generate predictors
  X_obs <- do.call(rbind,lapply(1:n,function(ii){
    Z_s <- rep(rnorm(3),each=5)
    omega_s <- rnorm(15,sd=.1)
    c(Z_s + omega_s,rnorm(15))
  }))
  X_train <- X_obs[1:n_train,]
  X_test <- X_obs[n_train+1:n_test,]
  
  # generate outcomes
  y_train <- c(X_train %*% beta_true) + rnorm(n_train,sd=sigma_true)
  y_test <- c(X_test %*% beta_true) + rnorm(n_test,sd=sigma_true)
  
  df3[[s]] <- list(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  
}

save(df3, file = "./analyses_R1/df3.RData")

# condition 3 of Van Erp et al. (2019), but with smaller training set
set.seed(123)
beta_true <- c(rep(3,15),rep(0,15))
k <- length(beta_true)
sigma_true <- sqrt(225)
n_train <- 40
n_test <- 400
n <- n_train+n_test
reps <- 500
df4 <- vector(mode='list', length=reps)

for(s in 1:reps){
  
  # generate predictors
  X_obs <- do.call(rbind,lapply(1:n,function(ii){
    Z_s <- rep(rnorm(3),each=5)
    omega_s <- rnorm(15,sd=.1)
    c(Z_s + omega_s,rnorm(15))
  }))
  X_train <- X_obs[1:n_train,]
  X_test <- X_obs[n_train+1:n_test,]
  
  # generate outcomes
  y_train <- c(X_train %*% beta_true) + rnorm(n_train,sd=sigma_true)
  y_test <- c(X_test %*% beta_true) + rnorm(n_test,sd=sigma_true)
  
  df4[[s]] <- list(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  
}
save(df4, file = "./analyses_R1/df4.RData")


# condition 5 of Van Erp et al. (2019)
set.seed(123)
beta_true <- c(rep(3,10),rep(0,10),rep(3,10))
k <- length(beta_true)
sigma_true <- sqrt(225)
n_train <- 40
n_test <- 400
n <- n_train+n_test
reps <- 500
df5 <- vector(mode='list', length=reps)

for(s in 1:reps){
  
  # generate predictors
  X_obs <- do.call(rbind,lapply(1:n,function(ii){
    Z_s <- rep(rnorm(3),each=5)
    omega_s <- rnorm(15,sd=.1)
    c(Z_s + omega_s,rnorm(15))
  }))
  X_train <- X_obs[1:n_train,]
  X_test <- X_obs[n_train+1:n_test,]
  
  # generate outcomes
  y_train <- c(X_train %*% beta_true) + rnorm(n_train,sd=sigma_true)
  y_test <- c(X_test %*% beta_true) + rnorm(n_test,sd=sigma_true)
  
  df5[[s]] <- list(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  
}
save(df5, file = "./analyses_R1/df5.RData")

# condition 3 of Van Erp et al. (2019) with just 5 non-zero effects
set.seed(123)
beta_true <- c(rep(3,5),rep(0,25))
k <- length(beta_true)
sigma_true <- sqrt(225)
n_train <- 200
n_test <- 400
n <- n_train+n_test
reps <- 500
df6 <- vector(mode='list', length=reps)

for(s in 1:reps){
  
  # generate predictors
  X_obs <- do.call(rbind,lapply(1:n,function(ii){
    Z_s <- rep(rnorm(3),each=5)
    omega_s <- rnorm(15,sd=.1)
    c(Z_s + omega_s,rnorm(15))
  }))
  X_train <- X_obs[1:n_train,]
  X_test <- X_obs[n_train+1:n_test,]
  
  # generate outcomes
  y_train <- c(X_train %*% beta_true) + rnorm(n_train,sd=sigma_true)
  y_test <- c(X_test %*% beta_true) + rnorm(n_test,sd=sigma_true)
  
  df6[[s]] <- list(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  
}

save(df6, file = "./analyses_R1/df6.RData")

# Based on the results, two conditions are added with an intermediate sample size for condition 1 and 3
# to determine when ABR breaks exactly

# condition 1 of Van Erp et al. (2019) with n_train of 20
set.seed(123)
beta_true <- c(3,1.5,0,0,2,0,0,0)
k <- length(beta_true)
sigma_true <- 3
n_train <- 20
n_test <- 200
n <- n_train+n_test
reps <- 500
df7 <- vector(mode='list', length=reps)

for(s in 1:reps){
  
  # generate predictors
  X_obs <- matrix(rnorm(n*k),ncol=k)
  X_train <- X_obs[1:n_train,]
  X_test <- X_obs[n_train+1:n_test,]
  
  # generate outcomes
  y_train <- c(X_train %*% beta_true) + rnorm(n_train,sd=sigma_true)
  y_test <- c(X_test %*% beta_true) + rnorm(n_test,sd=sigma_true)
  
  df7[[s]] <- list(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  
}

save(df7, file = "./analyses_R1/df7.RData")

# condition 3 of Van Erp et al. (2019) with n_train of 100
set.seed(123)
beta_true <- c(rep(3,15),rep(0,15))
k <- length(beta_true)
sigma_true <- sqrt(225)
n_train <- 100
n_test <- 400
n <- n_train+n_test
reps <- 500
df8 <- vector(mode='list', length=reps)

for(s in 1:reps){
  
  # generate predictors
  X_obs <- do.call(rbind,lapply(1:n,function(ii){
    Z_s <- rep(rnorm(3),each=5)
    omega_s <- rnorm(15,sd=.1)
    c(Z_s + omega_s,rnorm(15))
  }))
  X_train <- X_obs[1:n_train,]
  X_test <- X_obs[n_train+1:n_test,]
  
  # generate outcomes
  y_train <- c(X_train %*% beta_true) + rnorm(n_train,sd=sigma_true)
  y_test <- c(X_test %*% beta_true) + rnorm(n_test,sd=sigma_true)
  
  df8[[s]] <- list(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
  
}

save(df8, file = "./analyses_R1/df8.RData")

##### Analyze with exact regularization
priors <- c("ridge","lasso","horseshoe") 
nreps <- 1:500
#cond <- 1:6
cond <- 7
sett <- expand.grid("prior"=priors, 
                    "nrep"=nreps, 
                    "cond"=cond)

priors <- c("group_lasso") 
nreps <- 1:500
#cond <- 3:6
cond <- 8
sett <- expand.grid("prior"=priors, 
                    "nrep"=nreps, 
                    "cond"=cond)

sim_fun <- function(sett, pos, df7){
  
  prior <- sett$prior[pos]
  nrep <- sett$nrep[pos]
  cond <- sett$cond[pos]

  if(cond == 1){
    df <- df1[[nrep]]
  } else if(cond == 2){
    df <- df2[[nrep]]
  } else if(cond == 3){
    df <- df3[[nrep]]
  } else if(cond == 4){
    df <- df4[[nrep]]
  } else if(cond == 5){
    df <- df5[[nrep]]
  } else if(cond == 6){
    df <- df6[[nrep]]
  } else if(cond == 7){
    df <- df7[[nrep]]
  } else if(cond == 8){
    df <- df8[[nrep]]
  }
  
  # exact regularized fit with stan
  standat <- list(N_train=nrow(df$X_train), p=ncol(df$X_train), y_train=c(df$y_train), X_train=df$X_train,
                  N_test=nrow(df$X_test), X_test=df$X_test)
  modFB <- stan_model(paste0("./analyses_R1/", prior, ".stan"))
  fit.mcmc <- sampling(modFB, data=standat, iter=4000, chains = 4, control=list(adapt_delta=0.999, stepsize=0.001, max_treedepth=20))
  summ_mcmc <- summary(fit.mcmc)$summary
  save(summ_mcmc, file = paste0("./analyses_R1/summaries/cond", cond, "_rep", nrep, "_", prior, ".RData"))
  
  # compute PMSE
  # program got stuck here, so computed the outcome later based on the summaries
  # y_test_gen <- summ_mcmc[grep("y_test",rownames(summ_mcmc)),c("mean")]
  #  PMSE <- mean((y_test_gen - y_test)^2)
  
 # out <- data.frame("cond" = cond, "prior" = prior, "nrep" = nrep, "PMSE" = PMSE)
}
  
nworkers <- 6 # number of cores to use, max = 40
cl <- parallel::makeCluster(nworkers, type = "FORK")# create cluster
clusterSetRNGStream(cl, 123) 
out <- clusterApplyLB(cl, 1:nrow(sett), sim_fun, sett = sett, df7=df7)
stopCluster(cl) 
#out_df <- do.call(rbind, out)

#save(out_df,file="./analyses_R1/PMSE_exact.RData")

## Combine output
fls <- list.files("./analyses_R1/summaries")
PMSE <- matrix(NA, nrow = length(fls), ncol = 4)
colnames(PMSE) <- c("cond", "rep", "prior", "PMSE")
for(i in 1:length(fls)){
  load(paste0("./analyses_R1/summaries/", fls[i]))
  nm <- strsplit(fls[i], "_")
  cond <- as.numeric(strsplit(nm[[1]][1], "cond")[[1]][2])
  PMSE[i, 1] <- cond
  nrep <- as.numeric(strsplit(nm[[1]][2], "rep")[[1]][2])
  PMSE[i, 2] <- nrep
  prior <- strsplit(nm[[1]][3], ".RData")[[1]][1]
  PMSE[i, 3] <- prior
  
  if(cond == 1){
    df <- df1[[nrep]]
  } else if(cond == 2){
    df <- df2[[nrep]]
  } else if(cond == 3){
    df <- df3[[nrep]]
  } else if(cond == 4){
    df <- df4[[nrep]]
  } else if(cond == 5){
    df <- df5[[nrep]]
  } else if(cond == 6){
    df <- df6[[nrep]]
  } else if(cond == 7){
    df <- df7[[nrep]]
  } else if(cond == 8){
    df <- df8[[nrep]]
  }
  
  # compute PMSE
  y_test_gen <- summ_mcmc[grep("y_test",rownames(summ_mcmc)),c("mean")]
  y_test <- df$y_test
  PMSE[i, 4] <- mean((y_test_gen - y_test)^2)
}

save(PMSE, file="./analyses_R1/PMSE_exact_cond78.RData")

## Combine
load("./analyses_R1/PMSE_exact_cond78.RData")
pmse78 <- PMSE

load("./analyses_R1/PMSE_exact_full.RData")
PMSE <- rbind.data.frame(PMSE, pmse78)

save(PMSE, file = "./analyses_R1/PMSE_exact_complete.RData")

## Final results
load("./analyses_R1/PMSE_exact_complete.RData")

PMSE$PMSE <- as.numeric(PMSE$PMSE)
res <- PMSE %>% 
  group_by(cond, prior) %>% 
  summarise(median = median(PMSE), sd = sd(PMSE)) 

# add bootstrapped SEs like in Van Erp et al. (2019)
N <- 500 # number of median MSEs to compute
n <- 500 # number of MSEs to draw each time
set.seed(123)
res <- PMSE %>%
  group_by(cond, prior) %>% 
  mutate(boot_sd = sd(replicate(N, median(sample(PMSE, n, replace=T))))) %>% 
  summarise(median = median(PMSE), sd = sd(PMSE), sd_boot = mean(boot_sd))
res

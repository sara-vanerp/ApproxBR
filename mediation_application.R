# Mediation application using approximate Bayesian regularization
## Author: Sara van Erp

# Note 
# Data can be downloaded from: https://www.ebi.ac.uk/arrayexpress/experiments/E-GEOD-77445/
# Similar preprocessing steps have been taken as in van Kesteren and Oberski (2019):
# X, Y and M have been residualized with respect to their intercept, age and sex and the top 45
# potential mediators (out of 385882) have been selected in terms of their absolute product of correlations with X and Y.
# Preprocessing was done using code provided by Erik-Jan van Kesteren.
# In addition to the top 60 potential mediators, the 5 mediators selected in van Kesteren and Oberski (2019) through the
# CMF algorithm were manually added as well, resulting in a total of 65 potential mediators.

library(lavaan)
library(shrinkem)
library(blavaan)
library(rstan)
options(mc.cores = 4)
rstan_options(auto_write = TRUE)
library(tidyr)
library(MCMCglmm)
library(ggplot2)
library(bayesplot)

seed <- 11052023
set.seed(seed)

##### Data preparation ------
load("./data/methylation_preproc.RData")

# split into 90% training and 10% test set
ntrain <- round(0.9*nrow(x))
ntest <- nrow(x) - ntrain

# scale
xtrain <- scale(x[1:ntrain, ])
ytrain <- scale(y[1:ntrain], )
mtrain <- scale(MselAB[1:ntrain, ])

xtest <- scale(x[(ntrain+1):nrow(x), ])
ytest <- scale(y[(ntrain+1):nrow(x)], )
mtest <- scale(MselAB[(ntrain+1):nrow(x), ])

# create data for lavaan
nM <- ncol(MselAB)
lav_train <- cbind.data.frame(xtrain, ytrain, mtrain)
colnames(lav_train) <- c("X", "Y", paste0("M", 1:nM))

lav_test <- cbind.data.frame(xtest, ytest, mtest)
colnames(lav_test) <- c("X", "Y", paste0("M", 1:nM))

##### Analyses ------

# Option 1: Approximate with shrinkem (ridge and horseshoe priors)
# because we want to regularize the indirect effects (ab), we need to first
# fit the model with blavaan using uninformative priors to get the MLE and error cov matrix

# model specification
M <- paste0("M", 1:nM, " ~ ", "X \n")
Y <- c("Y ~ ", paste0("M", 1:nM, sep = " + "), "X \n")
lavmod <- c(M, Y)

defaultPriors <- dpriors(beta = "normal(0, 10000)") 

fit_blav <- bsem(lavmod, 
                 data = lav_train, 
                 std.lv = TRUE, 
                 dp = defaultPriors,
                 bcontrol = list(cores = 3),
                 seed = seed)
save(fit_blav, file = "./results/fitobjects/fit_blav_uninformative_mediation.RData")

load("./results/fitobjects/fit_blav_uninformative_mediation.RData")
draws <- as.matrix(blavInspect(fit_blav, what = "mcmc"))

# compute indirect effects
draws_ab <- matrix(NA, nrow = nrow(draws), ncol = nM)
for(i in 1:nM){
  draws_a <- draws[, grep(paste0("M", i, "~X"), colnames(draws), fixed = TRUE)]
  draws_b <- draws[, grep(paste0("Y~M", i, "$"), colnames(draws))]
  draws_ab[, i] <- draws_a * draws_b
}

mle <- colMeans(draws_ab)
covmat <- cov(draws_ab)

# shrinkem with ridge prior
shrink_ridge <- shrinkem(mle, covmat, type="ridge", iterations = 4000)
save(shrink_ridge, file = "./results/fitobjects/fit_shrinkem_ridge_mediation.RData")

# shrinkem with horseshoe prior
shrink_hs <- shrinkem(mle, covmat, type="horseshoe", iterations = 4000)
save(shrink_hs, file = "./results/fitobjects/fit_shrinkem_hs_mediation.RData")

# Option 2: Exact with blavaan and stan (ridge and horseshoe priors)
# ridge with blavaan
# for some reason, a manual ridge implementation directly on the indirect effect in stan is not converging
# use blavaan instead with shrinkage on the direct paths
# extract estimated prior variance for use in blavaan
d <- shrink_ridge$draws
lam2 <- d$lambda2
lam <- sqrt(lam2)
mean(lam) # 0.006

# estimated prior sd shrinkem = 0.006 but this is the prior on the indirect effect ab
# blavaan specifies the ridge on the direct effects a & b
# using a prior sd of mean(lam) gives a product prior on the indirect effect of mean(lam)*mean(lam)
# however, this number is so small it becomes tricky for stan to sample and takes really long, so instead use a prior sd of 0.01
ridgePriors <- dpriors(beta = "normal(0, 0.01)")

fit_blav <- bsem(lavmod, 
                 data = lav_train, 
                 std.lv = TRUE, 
                 dp = ridgePriors,
                 bcontrol = list(cores = 3),
                 seed = seed,
                 sample = 4000)
save(fit_blav, file = "./results/fitobjects/fit_blavaan_ridge_mediation.RData")

# horseshoe
standat <- list(N = nrow(lav_train),
                nM = ncol(lav_train)-2,
                X = lav_train$X,
                M = lav_train[, grep("M", colnames(lav_train))],
                Y = lav_train$Y,
                s0 = 1)
mod <- stan_model("./models/exact_mediation_horseshoe.stan")
fit <- sampling(mod, data = standat, iter = 8000)
save(fit, file = "./results/fitobjects/fit_exact_hs_mediation.RData")

##### Results: Estimation ------
# shrinkem
load("./results/fitobjects/fit_shrinkem_ridge_mediation.RData")
res1 <- cbind.data.frame(rownames(shrink_ridge$estimates),
                         shrink_ridge$estimates[, c('shrunk.mean', 'shrunk.mode', 'shrunk.lower', 'shrunk.upper')],
                         "shrinkem", "ridge")
colnames(res1) <- c("par", "mean", "mode", "ci.lower", "ci.upper", "package", "prior")
res1$par <- paste0("beta", 1:50)

load("./results/fitobjects/fit_shrinkem_hs_mediation.RData")
res2 <- cbind.data.frame(rownames(shrink_hs$estimates),
                         shrink_hs$estimates[, c('shrunk.mean', 'shrunk.mode', 'shrunk.lower', 'shrunk.upper')],
                         "shrinkem", "hs")
colnames(res2) <- c("par", "mean", "mode", "ci.lower", "ci.upper", "package", "prior")
res1$par <- paste0("beta", 1:50)

# blavaan
load("./results/fitobjects/fit_blavaan_ridge_mediation.RData")
draws_blav <- as.matrix(blavInspect(fit_blav, what = "mcmc"))
# compute indirect effects
draws_ab <- matrix(NA, nrow = nrow(draws_blav), ncol = nM)
for(i in 1:nM){
  draws_a <- draws_blav[, grep(paste0("M", i, "~X"), colnames(draws_blav), fixed = TRUE)]
  draws_b <- draws_blav[, grep(paste0("Y~M", i, "$"), colnames(draws_blav))]
  draws_ab[, i] <- draws_a * draws_b
}

res3 <- data.frame(colMeans(draws_ab))
res3$par <- paste0("beta", 1:nM)
res3$LB <- apply(draws_ab, 2, function(x) quantile(x, 0.025))
res3$UB <- apply(draws_ab, 2, function(x) quantile(x, 0.975))
res3$mode <- posterior.mode(draws_ab)
colnames(res3) <- c("mean", "par", "ci.lower", "ci.upper", "mode")
res3$package <- "blavaan"
res3$prior <- "ridge"

# stan
load("./results/fitobjects/fit_exact_hs_mediation.RData")
summ <- summary(fit, prob = c(0.025, 0.975))$summary
outsel <- summ[grep("ab\\[", rownames(summ)), c("mean", "2.5%", "97.5%")]
fit.mcmc <- as.matrix(fit)
modes <- posterior.mode(fit.mcmc)
modes.sel <- modes[grep("ab\\[", names(modes))]
outsel <- cbind(outsel, "mode" = modes.sel)
nms <- rownames(outsel)
res4 <- cbind.data.frame(nms, "hs", "exact", outsel)
colnames(res4) <- c("par", "prior", "package", "mean", "ci.lower", "ci.upper", "mode")
res4$par <- paste0("beta", 1:nM)

# combine
res <- rbind.data.frame(res1, res2, res3, res4)
save(res, file = "./results/full_results_mediation.Rdata")

load("./results/full_results_mediation.Rdata")

# Select variables based on 95% CI
res$sel <- NA
for(i in 1:nrow(res)){
  res$sel[i] <- ifelse(res$`ci.lower`[i] < 0 & res$`ci.upper`[i] > 0, 0, 1) # 1 if mediator is selected
}
res$meth <- paste(res$prior, res$package, sep = " ")

colnames(res) <- c("Variable", "Mean", "Mode", "LB", "UB", "Algorithm", "Prior", "Selected", "Method")

# Plot estimates
pd <- position_dodge(0.8)

# something weird is going on with the exact horseshoe
sel <- which(res$Variable %in% paste0("beta", 30:33))
ggplot(res[sel, ], aes(x = Mean, y = Variable, colour = Method, linetype = Method)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 3) +
  geom_point(aes(x = Mode), position = pd, size = 3, shape = 17) +
  scale_linetype_manual("", values = c(1, 2, 1, 2)) +
  scale_colour_manual("", values = c("black", "red", "blue", "green")) +
  ylab("Variable") + xlab("Posterior estimates and 95% CI") + theme_bw(base_size = 25) + 
  guides(colour = guide_legend(nrow = 2), linetype = guide_legend(nrow = 2)) +
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom", legend.key.width = unit(1.5, "cm")) 


load("./results/fitobjects/fit_exact_hs_mediation.RData")
posterior <- as.matrix(fit)
sel <- posterior[, grep("ab\\[", colnames(posterior))]
mcmc_areas(sel[, 31:32],
           prob = 0.95) 

# Remove the exact horseshoe from the results because those point estimates
# do not seem really reflective of the heavy-tailed posteriors
sel <- which(res$Method != "hs exact")
png(file = "./results/mediation_comparison_priors.png", width = 1000, height = 1300)
ggplot(res[sel, ], aes(x = Mean, y = Variable, colour = Method, linetype = Method)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 3) +
  geom_point(aes(x = Mode), position = pd, size = 3, shape = 17) +
  scale_linetype_manual("", values = c(1, 2, 1)) +
  scale_colour_manual("", values = c("black", "red", "blue")) +
  ylab("Variable") + xlab("Posterior estimates and 95% CI") + theme_bw(base_size = 25) + 
  guides(colour = guide_legend(nrow = 2), linetype = guide_legend(nrow = 2)) +
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom", legend.key.width = unit(1.5, "cm")) 
dev.off()

##### Results: Prediction ------
# compute PMSE
# for every draw of an indirect effect, add the direct effect
# multiply the total effect with the test X
# compute the difference with the test Y squared and take the mean over the Y values
# plot the predictive distribution or take its mean
pmse_fun <- function(draws_ind, dir, testX, testY){
  total = draws_ind + dir
  abX = as.matrix(total) %*% as.matrix(t(testX))
  predY = colSums(abX)
  pmse = mean((testY - predY)^2)
  return(pmse)
}

# use the direct effect from blavaan for all PMSEs
# note that the number of samples differs so use the mean estimate for the direct effect
load("./results/fitobjects/fit_blavaan_ridge_mediation.RData")
draws_blav <- as.matrix(blavInspect(fit_blav, what = "mcmc"))
draws_dir <- draws_blav[, grep("Y~X", colnames(draws_blav))]
dir <- mean(draws_dir)
# compute indirect effects
blav_ab <- matrix(NA, nrow = nrow(draws_blav), ncol = nM)
for(i in 1:nM){
  draws_a <- draws_blav[, grep(paste0("M", i, "~X"), colnames(draws_blav), fixed = TRUE)]
  draws_b <- draws_blav[, grep(paste0("Y~M", i, "$"), colnames(draws_blav))]
  blav_ab[, i] <- draws_a * draws_b
}

pmse_blav <- sapply(blav_ab, pmse_fun, dir = dir, testX = xtest, testY = ytest)
mean(pmse_blav)

load("./results/fitobjects/fit_shrinkem_ridge_mediation.RData")
draws_ridge <- shrink_ridge$draws$beta
pmse_ridge <- sapply(draws_ridge, pmse_fun, dir = dir, testX = xtest, testY = ytest)
mean(pmse_ridge)

load("./results/fitobjects/fit_shrinkem_hs_mediation.RData")
draws_hs <- shrink_hs$draws$beta
pmse_hs <- sapply(draws_hs, pmse_fun, dir = dir, testX = xtest, testY = ytest)
mean(pmse_hs)



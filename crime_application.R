# Crime application using approximate Bayesian regularization
## Author: Sara van Erp

library(rstan)
options(mc.cores = 4)
rstan_options(auto_write = TRUE)
library(shrinkem)
library(brms)
library(dplyr)
library(ggplot2)
library(MCMCglmm)

set.seed(07042023)

##### Data preparation ------
# Data can be downloaded from: https://archive.ics.uci.edu/ml/datasets/communities+and+crime+unnormalized
dat <- read.table("data/communities_crime_unnormalized.txt", sep=",", na.string="?")

head(dat)
colnames(dat) <- c("communityname", "state", "countyCode", "communityCode", "fold", "population", "householdsize", "racepctblack",
                   "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up",
                   "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst",
                   "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap",
                   "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed",
                   "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr",
                   "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids",
                   "PctWorkMom", "NumKidsBornNeverMar", "PctKidsBornNeverMar", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8",
                   "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell",
                   "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup",
                   "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos",
                   "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "OwnOccQrange",
                   "RentLowQ", "RentMedian", "RentHighQ", "RentQrange", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg",
                   "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85",
                   "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop",
                   "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian",
                   "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans",
                   "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop",
                   "murders", "murdPerPop", "rapes", "rapesPerPop", "robberies", "robbPerPop", "assaults", "assaultPerPop", "burglaries",
                   "burglPerPop", "larcenies", "larcPerPop", "autoTheft", "autoTheftPerPop", "arsons", "arsonsPerPop", "ViolentCrimesPerPop",
                   "nonViolPerPop")



# remove non-predictive attributes and possible prediction goals;
# keep only the total number of violent crimes to predict (others are subtotals) 
dat.sel <- subset(dat, select = -c(communityname, countyCode, communityCode, fold,
                                 murders, murdPerPop, rapes, rapesPerPop, robberies,
                                 robbPerPop, assaults, assaultPerPop, burglaries,
                                 burglPerPop, larcenies, larcPerPop, autoTheft,
                                 autoTheftPerPop, arsons, arsonsPerPop, nonViolPerPop))

# for simplicity, keep only the continuous predictors
# OwnOccQrange and RentQrange are removed too since these are functions of other predictors leading to singularities in the MLEs
df <- subset(dat.sel, select = -c(state, LemasGangUnitDeploy, OwnOccQrange, RentQrange))
summary(df)

# plot outcome measure 
hist(df$ViolentCrimesPerPop) # skewed

# log transform the outcome measure
hist(log(df$ViolentCrimesPerPop)) # more normal
df$ViolentCrimesPerPop <- log(df$ViolentCrimesPerPop)

# create design matrix
mod.mat <- stats::model.matrix(~., df)[, -1] # removes all NAs

# 90% training and 10% test set and standardize both
ntrain <- as.integer(0.9*nrow(mod.mat))
ntest <- nrow(mod.mat)-ntrain

train <- data.frame(scale(mod.mat[1:ntrain, ]))
test <- data.frame(scale(mod.mat[(ntrain+1):nrow(mod.mat), ]))

##### Analyses ------

# Option 1: Exact with Stan
input.dat <- list(N_train = nrow(train),
                  p = ncol(train)-1,
                  y_train = train$ViolentCrimesPerPop,
                  X_train = train[, -c(grep("ViolentCrimesPerPop", colnames(train)))])

# prior hyperparameters
s0 = 1
nu0 = 3

# ridge exact
standat <- c(input.dat, 
             list(s0 = s0,
                  nu0 = nu0))
mod <- stan_model("./models/exact_regression_ridge.stan")
fit <- sampling(mod, data = standat)
save(fit, file = paste0("./results/fitobjects/fit_exact_ridge_crime.RData"))

# lasso exact
standat <- c(input.dat, 
             list(s0 = s0,
                  nu0 = nu0))
mod <- stan_model("./models/exact_regression_lasso.stan")
fit <- sampling(mod, data = standat)
save(fit, file = paste0("./results/fitobjects/fit_exact_lasso_crime.RData"))
  
# horseshoe exact  
standat <- c(input.dat, 
             list(s0 = s0))
mod <- stan_model("./models/exact_regression_hs.stan")
fit <- sampling(mod, data = standat, iter = 8000) # results in divergences
save(fit, file = paste0("./results/fitobjects/fit_exact_hs_crime.RData"))

# Option 2: Approximate implementation in shrinkem

# get maximum likelihood estimates
lmfit <- lm(train$ViolentCrimesPerPop ~ -1 + ., train)
summary(lmfit)

# extract MLEs
mle <- coef(lmfit)
covmat <- vcov(lmfit)

# ridge approximation shrinkem
shrink.ridge <- shrinkem(mle, Sigma = covmat, type = "ridge")
save(shrink.ridge, file = "./results/fitobjects/fit_shrinkem_ridge_crime.RData")

# lasso approximation shrinkem
shrink.lasso <- shrinkem(mle, Sigma = covmat, type = "lasso")
save(shrink.lasso, file = "./results/fitobjects/fit_shrinkem_lasso_crime.RData")

# horseshoe approximation shrinkem
shrink.hs <- shrinkem(mle, Sigma = covmat, type = "horseshoe") 
save(shrink.hs, file = "./results/fitobjects/fit_shrinkem_hs_crime.RData")

# Option 3: Approximate implementation in Stan
# note: this option is not included in the manuscript but only an illustration
# of how to implement the approximate model in Stan (models can be adapted easily for different priors)

# prior hyperparameters: same as for the exact implementation
s0 = 1
nu0 = 3

# ridge approximation Stan
standat <- list(p = length(mle),
                mle = mle,
                errorcov = covmat,
                s0 = s0,
                nu0 = nu0)

mod <- stan_model("./models/approx_regression_ridge.stan")
fit <- sampling(mod, data = standat)
save(fit, file = paste0("./results/fitobjects/fit_approxStan_ridge_crime.RData"))

# lasso approximation Stan 
standat <- list(p = length(mle),
                mle = mle,
                errorcov = covmat,
                s0 = s0,
                nu0 = nu0)

mod <- stan_model("./models/approx_regression_lasso.stan")
fit <- sampling(mod, data = standat)
save(fit, file = paste0("./results/fitobjects/fit_approxStan_lasso_crime.RData"))

# horseshoe approximation Stan 
standat <- list(p = length(mle),
                mle = mle,
                errorcov = covmat,
                s0 = s0)

mod <- stan_model("./models/approx_regression_hs.stan")
fit <- sampling(mod, data = standat, iter = 8000) # note the divergences and convergence warnings here
save(fit, file = paste0("./results/fitobjects/fit_approxStan_hs_crime.RData"))

##### Results: Estimation ------
# add posterior modes and combine results approximate and exact Stan implementations
get.results <- function(fitobj, prior, algorithm, nms = names(mle)){
  summ <- summary(fitobj, prob = c(0.025, 0.975))$summary
  if(algorithm == "approx"){
    outsel <- summ[grep("theta\\[", rownames(summ)), c("mean", "2.5%", "97.5%")]
  } else if(algorithm == "exact"){
    outsel <- summ[grep("beta\\[", rownames(summ)), c("mean", "2.5%", "97.5%")]
  }
  
  # add posterior modes
  fit.mcmc <- as.matrix(fitobj)
  modes <- posterior.mode(fit.mcmc)
  if(algorithm == "approx"){
    modes.sel <- modes[grep("theta\\[", names(modes))]
  } else if(algorithm == "exact"){
    modes.sel <- modes[grep("beta\\[", names(modes))]
  }
  outsel <- cbind(outsel, "mode" = modes.sel)
  
  res <- cbind.data.frame(nms, prior, algorithm, outsel)
  return(res)
}

load("./results/fitobjects/fit_approxStan_ridge_crime.RData")
res.ridge1 <- get.results(fitobj = fit, prior = "ridge", algorithm = "approx")
load("./results/fitobjects/fit_approxStan_lasso_crime.RData")
res.lasso1 <- get.results(fitobj = fit, prior = "lasso", algorithm = "approx")
load("./results/fitobjects/fit_approxStan_hs_crime.RData")
res.hs1 <- get.results(fitobj = fit, prior = "hs", algorithm = "approx")

load("./results/fitobjects/fit_exact_ridge_crime.RData")
res.ridge2 <- get.results(fitobj = fit, prior = "ridge", algorithm = "exact")
load("./results/fitobjects/fit_exact_lasso_crime.RData")
res.lasso2 <- get.results(fitobj = fit, prior = "lasso", algorithm = "exact")
load("./results/fitobjects/fit_exact_hs_crime.RData")
res.hs2 <- get.results(fitobj = fit, prior = "hs", algorithm = "exact")

res <- rbind.data.frame(res.ridge1, res.lasso1, res.hs1,
                        res.ridge2, res.lasso2, res.hs2)

# select variables based on 95% CI 
sel <- rep(NA, nrow(res))
for(i in 1:nrow(res)){
  sel[i] <- ifelse(res$`2.5%`[i] <= 0 & res$`97.5%`[i] >=0, FALSE, TRUE)
}

res$select <- sel

head(res)
colnames(res) <- c("Variable", "Prior", "Algorithm", "Mean", "LB", "UB", "Mode", "Included")

# add results shrinkem
load("./results/fitobjects/fit_shrinkem_ridge_crime.RData")
load("./results/fitobjects/fit_shrinkem_lasso_crime.RData")
load("./results/fitobjects/fit_shrinkem_hs_crime.RData")

## shrinkem
res.ridge <- summary(shrink.ridge)
res.ridge <- res.ridge[, c(grep("shrunk.mean|shrunk.mode|shrunk.lower|shrunk.upper|nonzero", colnames(res.ridge)))]
colnames(res.ridge) <- c("Mean", "Mode", "LB", "UB", "Included")
res.ridge$Variable <- rownames(res.ridge)
res.ridge$Prior <- "ridge"
res.ridge$Algorithm <- "shrinkem"

res.lasso <- summary(shrink.lasso)
res.lasso <- res.lasso[, c(grep("shrunk.mean|shrunk.mode|shrunk.lower|shrunk.upper|nonzero", colnames(res.lasso)))]
colnames(res.lasso) <- c("Mean", "Mode", "LB", "UB", "Included")
res.lasso$Variable <- rownames(res.lasso)
res.lasso$Prior <- "lasso"
res.lasso$Algorithm <- "shrinkem"

res.hs <- summary(shrink.hs)
res.hs <- res.hs[, c(grep("shrunk.mean|shrunk.mode|shrunk.lower|shrunk.upper|nonzero", colnames(res.hs)))]
colnames(res.hs) <- c("Mean", "Mode", "LB", "UB", "Included")
res.hs$Variable <- rownames(res.hs)
res.hs$Prior <- "hs"
res.hs$Algorithm <- "shrinkem"

res.shrink <- rbind.data.frame(res.ridge, res.lasso, res.hs)

res <- rbind.data.frame(res, res.shrink)
save(res, file = "./results/full_results_crime.RData")

# Visualize estimates
load("./results/full_results_crime.RData")

# Compare CIs and different priors for largest effects
pd <- position_dodge(0.8)
# reorder predictors based on estimates horseshoe
sel <- res[which(res$Prior == "hs" & res$Algorithm == "exact"), ]
ord <- sel[order(abs(sel$Mean), decreasing = TRUE), "Variable"]
res$Variable <- factor(res$Variable, levels = ord)

# ridge
df.sel <- res[which(res$Prior %in% c("ridge", "lasso", "hs") & res$Variable %in% ord[c(1:10, 111:121)] & res$Algorithm %in% c("exact", "shrinkem")), ]
df.sel$Prior <- factor(df.sel$Prior)
levels(df.sel$Prior) <- list("Horseshoe" = "hs",
                             "Lasso" = "lasso",
                             "Ridge" = "ridge")
df.sel$Method <- paste(df.sel$Prior, df.sel$Algorithm, sep =" ")
png(file = "./results/crime_comparison_priors.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Mean, y = Variable, colour = Method, linetype = Method)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 3) +
  geom_point(aes(x = Mode), position = pd, size = 3, shape = 17) +
  scale_linetype_manual("", values = c(1, 2, 1, 2, 1, 2)) +
  scale_colour_manual("", values = c("blue", "blue", "red", "red", "black", "black")) + 
  ylab("Variable") + xlab("Posterior estimates and 95% CI") + theme_bw(base_size = 25) + 
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom", legend.key.width = unit(1.5, "cm"))
dev.off()

# check where difference in posterior mean and mode for the exact horseshoe comes from
load("./results/fitobjects/fit_exact_hs_crime.RData")
mcmc_areas(as.matrix(fit), pars = "beta[45]")

##### Results: PMSE ------
# PMSE is computed based on unselected estimates
# 95% interval is not ideal to select predictors, so I expect this to worsen the PMSE

testX <- as.data.frame(t(test[, -grep("ViolentCrimesPerPop", colnames(test))]))
testX$Variable <- rownames(testX)
testY <- test$ViolentCrimesPerPop

res$Method <- factor(paste(res$Prior, res$Algorithm, sep = "_"))
out <- data.frame(NA)
for(i in 1:length(levels(res$Method))){
  sel <- res[which(res$Method == levels(res$Method)[i]), c("Variable", "Mean")]
  comb <- merge(sel, testX, by = "Variable")
  
  test.obs <- comb[, -c(grep("Variable|Mean", colnames(comb)))]
  est <- comb$Mean
  predY <- apply(test.obs, 2, function(x) sum(est*x))
  pmse <- mean((testY - predY)^2)
  
  out[i, 1] <- levels(res$Method)[i]
  out[i, 2] <- pmse
}

colnames(out) <- c("Method", "PMSE")
print(out, digits = 2)

# add MSE regular lm
sel <- data.frame("Estimate" = lmfit$coefficients,
                  "Variable" = names(lmfit$coefficients))
comb <- merge(sel, testX, by = "Variable")
test.obs <- comb[, -c(grep("Variable|Estimate", colnames(comb)))]
est <- comb$Estimate
predY <- apply(test.obs, 2, function(x) sum(est*x))
pmse <- mean((testY - predY)^2)
print(pmse, digits = 2)

##### Results: Number of selected variables -----
head(res)
df.sel <- res[which(res$Prior %in% c("ridge", "lasso", "hs") & res$Algorithm %in% c("exact", "shrinkem")), ]
df.sel$Method <- paste(df.sel$Prior, df.sel$Algorithm, sep = "_")

df.sel %>% 
  group_by(Method) %>% 
  summarize(sum = sum(Included))




##### Analyses revision 1 -----
## Visualize the results on a diagonal (in line with other figures)
load("./results/full_results_crime.RData")

approx <- res[which(res$Algorithm == "shrinkem"), ]
colnames(approx) <- c("Variable", "Prior", "Algorithm", 
                      "Approx. mean", "Approx. LB", "Approx. UB",
                      "Approx. mode", "Approx. incl.")
exact <- res[which(res$Algorithm == "exact"), ]
colnames(exact) <- c("Variable", "Prior", "Algorithm",
                     "Exact mean", "Exact LB", "Exact UB",
                     "Exact mode", "Exact incl.")

plotdat <- merge(approx, exact, by=c("Variable", "Prior"))
plotdat$Prior <- plyr::revalue(plotdat$Prior, 
                             c("hs" = "Horseshoe",
                               "lasso" = "Lasso",
                               "ridge" = "Ridge"))

png(file = "./results/points_intervals_diagonal_crime.png", width = 1400, height = 800)
ggplot(plotdat, aes(x = `Exact mean`, y = `Approx. mean`)) +
  facet_wrap("Prior") +
  geom_point() +
  coord_cartesian(xlim = c(-1, 1), ylim = c(-1, 1)) +
  geom_abline(intercept = 0, slope = 1, lty = 2) +
  geom_linerange(aes(ymin = `Approx. LB`, ymax = `Approx. UB`)) +
  geom_linerange(aes(xmin = `Exact LB`, xmax = `Exact UB`)) +
  xlab("Exact") + ylab("Approximate") +
  theme_bw(base_size = 25)
dev.off()

## Visualize histograms of the modes
load("./results/full_results_crime.RData")

res <- res[which(res$Algorithm != "approx"), ] # remove approximate implementation stan

res_ml <- data.frame("Variable" = names(coef(lmfit)),
                     "Prior" = "Unregularized",
                     "Algorithm" = "exact",
                     "Mean" = coef(lmfit),
                     "LB" = NA,
                     "UB" = NA,
                     "Mode" = coef(lmfit),
                     "Included" = NA)

res_ml <- res_ml[-which(res_ml$Mode == max(res_ml$Mode)), ] # Remove extreme estimate
res_ml <- res_ml[-which(res_ml$Mode == min(res_ml$Mode)), ] # Remove extreme estimate

res <- rbind.data.frame(res, res_ml)

res$Prior <- plyr::revalue(res$Prior, 
                               c("hs" = "Horseshoe",
                                 "lasso" = "Lasso",
                                 "ridge" = "Ridge"))

png(file = "./results/hist_mode_crime.png", width = 1000, height = 800)
ggplot(res, aes(x = `Mode`)) +
  geom_histogram() +
  facet_grid(`Algorithm`~`Prior`) +
  theme_bw(base_size = 25) + ylab("")
dev.off()

# check modes within 0.1 of 0
res %>%
  group_by(Prior, Algorithm) %>%
  summarize(sum(`Mode` > -0.1 & `Mode` < 0.1))

##### Additional cross-validation on the full data and half of the data -----
## Note: 25% of the data is not possible because then ML estimates are no longer available

## Use 10-fold CV to compute the PMSE and visualize with a boxplot
K <- 10
nrow(mod.mat)/K

mod.mat <- mod.mat[sample(nrow(mod.mat), replace = FALSE), ]
fold <- list()
fold[[1]] <- data.frame(scale(mod.mat[1:32, ]))
fold[[2]] <- data.frame(scale(mod.mat[33:64, ]))
fold[[3]] <- data.frame(scale(mod.mat[65:96, ]))
fold[[4]] <- data.frame(scale(mod.mat[97:128, ]))
fold[[5]] <- data.frame(scale(mod.mat[129:160, ]))
fold[[6]] <- data.frame(scale(mod.mat[161:192, ]))
fold[[7]] <- data.frame(scale(mod.mat[193:224, ]))
fold[[8]] <- data.frame(scale(mod.mat[225:256, ]))
fold[[9]] <- data.frame(scale(mod.mat[257:288, ]))
fold[[10]] <- data.frame(scale(mod.mat[289:nrow(mod.mat), ]))

# Run the analysis on k-1 folds using the remaining fold as test set
out <- vector(mode = "list", length = K)
for(k in 1:K){
  test <- fold[[k]]
  fold_train <- fold
  fold_train[[k]] <- NULL
  train <- do.call(rbind.data.frame, fold_train)
  
  # Option 1: Exact with Stan
  input.dat <- list(N_train = nrow(train),
                    p = ncol(train)-1,
                    y_train = train$ViolentCrimesPerPop,
                    X_train = train[, -c(grep("ViolentCrimesPerPop", colnames(train)))])
  
  # prior hyperparameters
  s0 = 1
  nu0 = 3
  
  # ridge exact
  standat <- c(input.dat, 
               list(s0 = s0,
                    nu0 = nu0))
  mod <- stan_model("./models/exact_regression_ridge.stan")
  fit.ridge <- sampling(mod, data = standat)
  
  # lasso exact
  standat <- c(input.dat, 
               list(s0 = s0,
                    nu0 = nu0))
  mod <- stan_model("./models/exact_regression_lasso.stan")
  fit.lasso <- sampling(mod, data = standat)
  
  # horseshoe exact  
  standat <- c(input.dat, 
               list(s0 = s0))
  mod <- stan_model("./models/exact_regression_hs.stan")
  fit.hs <- sampling(mod, data = standat, iter = 8000) 
  
  # Option 2: Approximate implementation in shrinkem
  
  # get maximum likelihood estimates
  lmfit <- lm(train$ViolentCrimesPerPop ~ -1 + ., train)
  
  # extract MLEs
  mle <- coef(lmfit)
  covmat <- vcov(lmfit)
  
  # ridge approximation shrinkem
  shrink.ridge <- shrinkem(mle, Sigma = covmat, type = "ridge", iterations = 5000)
  
  # lasso approximation shrinkem
  shrink.lasso <- shrinkem(mle, Sigma = covmat, type = "lasso", iterations = 5000)
  
  # horseshoe approximation shrinkem
  shrink.hs <- shrinkem(mle, Sigma = covmat, type = "horseshoe", iterations = 5000) 
  
  # Combine results exact algorithm
  get.results <- function(fitobj, prior, algorithm, nms = names(mle)){
    summ <- summary(fitobj, prob = c(0.025, 0.975))$summary
    if(algorithm == "approx"){
      outsel <- summ[grep("theta\\[", rownames(summ)), c("mean", "2.5%", "97.5%")]
    } else if(algorithm == "exact"){
      outsel <- summ[grep("beta\\[", rownames(summ)), c("mean", "2.5%", "97.5%")]
    }
    
    res <- cbind.data.frame(nms, prior, algorithm, outsel)
    return(res)
  }
  
  res.ridge1 <- get.results(fitobj = fit.ridge, prior = "ridge", algorithm = "exact")
  res.lasso1 <- get.results(fitobj = fit.lasso, prior = "lasso", algorithm = "exact")
  res.hs1 <- get.results(fitobj = fit.hs, prior = "hs", algorithm = "exact")
  
  res.exact <- rbind.data.frame(res.ridge1, res.lasso1, res.hs1)
  
  ## add results shrinkem
  res.ridge <- summary(shrink.ridge)
  res.ridge <- res.ridge[, c(grep("shrunk.mean|shrunk.lower|shrunk.upper", colnames(res.ridge)))]
  colnames(res.ridge) <- c("Mean", "LB", "UB")
  res.ridge$Variable <- rownames(res.ridge)
  res.ridge$Prior <- "ridge"
  res.ridge$Algorithm <- "shrinkem"
  
  res.lasso <- summary(shrink.lasso)
  res.lasso <- res.lasso[, c(grep("shrunk.mean|shrunk.lower|shrunk.upper", colnames(res.lasso)))]
  colnames(res.lasso) <- c("Mean", "LB", "UB")
  res.lasso$Variable <- rownames(res.lasso)
  res.lasso$Prior <- "lasso"
  res.lasso$Algorithm <- "shrinkem"
  
  res.hs <- summary(shrink.hs)
  res.hs <- res.hs[, c(grep("shrunk.mean|shrunk.lower|shrunk.upper", colnames(res.hs)))]
  colnames(res.hs) <- c("Mean", "LB", "UB")
  res.hs$Variable <- rownames(res.hs)
  res.hs$Prior <- "hs"
  res.hs$Algorithm <- "shrinkem"
  
  res.shrink <- rbind.data.frame(res.ridge, res.lasso, res.hs)
  colnames(res.exact) <- c("Variable", "Prior", "Algorithm", "Mean", "LB", "UB")
  res <- rbind.data.frame(res.exact, res.shrink)
  
  # Compute PMSE
  testX <- as.data.frame(t(test[, -grep("ViolentCrimesPerPop", colnames(test))]))
  testX$Variable <- rownames(testX)
  testY <- test$ViolentCrimesPerPop
  
  res$Method <- factor(paste(res$Prior, res$Algorithm, sep = "_"))
  out[[k]] <- data.frame(NA)
  for(i in 1:length(levels(res$Method))){
    sel <- res[which(res$Method == levels(res$Method)[i]), c("Variable", "Mean")]
    comb <- merge(sel, testX, by = "Variable")
    
    test.obs <- comb[, -c(grep("Variable|Mean", colnames(comb)))]
    est <- comb$Mean
    predY <- apply(test.obs, 2, function(x) sum(est*x))
    pmse <- mean((testY - predY)^2)
    
    out[[k]][i, 1] <- levels(res$Method)[i]
    out[[k]][i, 2] <- pmse
  }
  
  colnames(out[[k]]) <- c("Method", "PMSE")
  
  # add MSE regular lm
  sel <- data.frame("Estimate" = lmfit$coefficients,
                    "Variable" = names(lmfit$coefficients))
  comb <- merge(sel, testX, by = "Variable")
  test.obs <- comb[, -c(grep("Variable|Estimate", colnames(comb)))]
  est <- comb$Estimate
  predY <- apply(test.obs, 2, function(x) sum(est*x))
  pmse <- mean((testY - predY)^2)
  out[[k]][7, 1] <- "lm"
  out[[k]][7, 2] <- pmse
}

save(out, file ="./results/CV_PMSE_crime.RData")

pmse <- do.call(rbind.data.frame, out)
pmse$Method <- plyr::revalue(pmse$Method, 
                             c("hs_exact" = "Exact horseshoe",
                               "hs_shrinkem" = "App. horseshoe",
                               "lasso_exact" = "Exact lasso ",
                               "lasso_shrinkem" = "App. lasso",
                               "ridge_exact" = "Exact ridge",
                               "ridge_shrinkem" = "App. ridge",
                               "lm" = "Unregularized"))

png(file = "./results/CV_PMSE_crime.png", width = 1000, height = 800)
ggplot(pmse, aes(x = Method, y = PMSE)) +
  geom_boxplot() +
  scale_x_discrete(guide = guide_axis(angle = 90)) +
  theme_bw(base_size = 25)
dev.off()

## 10-fold CV PMSE for half of the data set
N <- round(0.5*nrow(mod.mat))
sel <- sample(1:nrow(mod.mat), N, replace = FALSE)
half_df <- mod.mat[sel, ]

## Use 10-fold CV to compute the PMSE and visualize with a boxplot
K <- 10
nrow(half_df)/K

# scaling the data per fold introduces NAs due to small variability in certain folds (specifically fold 5)
half_df <- data.frame(scale(half_df[sample(nrow(half_df), replace = FALSE), ])) # shuffle and scale data
fold <- list()
fold[[1]] <- half_df[1:16, ]
fold[[2]] <- half_df[17:(2*16), ]
fold[[3]] <- half_df[(2*16+1):(3*16), ]
fold[[4]] <- half_df[(3*16+1):(4*16), ]
fold[[5]] <- half_df[(4*16+1):(5*16), ]
fold[[6]] <- half_df[(5*16+1):(6*16), ]
fold[[7]] <- half_df[(6*16+1):(7*16), ]
fold[[8]] <- half_df[(7*16+1):(8*16), ]
fold[[9]] <- half_df[(8*16+1):(9*16), ]
fold[[10]] <- half_df[(9*16+1):nrow(half_df), ]

# Run the analysis on k-1 folds using the remaining fold as test set
out <- vector(mode = "list", length = K)
for(k in 1:K){
  test <- fold[[k]]
  fold_train <- fold
  fold_train[[k]] <- NULL
  train <- do.call(rbind.data.frame, fold_train)
  
  # Option 1: Exact with Stan
  input.dat <- list(N_train = nrow(train),
                    p = ncol(train)-1,
                    y_train = train$ViolentCrimesPerPop,
                    X_train = train[, -c(grep("ViolentCrimesPerPop", colnames(train)))])
  
  # prior hyperparameters
  s0 = 1
  nu0 = 3
  
  # ridge exact
  standat <- c(input.dat, 
               list(s0 = s0,
                    nu0 = nu0))
  mod <- stan_model("./models/exact_regression_ridge.stan")
  fit.ridge <- sampling(mod, data = standat)
  
  # lasso exact
  standat <- c(input.dat, 
               list(s0 = s0,
                    nu0 = nu0))
  mod <- stan_model("./models/exact_regression_lasso.stan")
  fit.lasso <- sampling(mod, data = standat)
  
  # horseshoe exact  
  standat <- c(input.dat, 
               list(s0 = s0))
  mod <- stan_model("./models/exact_regression_hs.stan")
  fit.hs <- sampling(mod, data = standat, iter = 8000) 
  
  # Option 2: Approximate implementation in shrinkem
  
  # get maximum likelihood estimates
  lmfit <- lm(train$ViolentCrimesPerPop ~ -1 + ., train)
  
  # extract MLEs
  mle <- coef(lmfit)
  covmat <- vcov(lmfit)
  
  # ridge approximation shrinkem
  shrink.ridge <- shrinkem(mle, Sigma = covmat, type = "ridge", iterations = 5000)
  
  # lasso approximation shrinkem
  shrink.lasso <- shrinkem(mle, Sigma = covmat, type = "lasso", iterations = 5000)
  
  # horseshoe approximation shrinkem
  shrink.hs <- shrinkem(mle, Sigma = covmat, type = "horseshoe", iterations = 5000) 
  
  # Combine results exact algorithm
  get.results <- function(fitobj, prior, algorithm, nms = names(mle)){
    summ <- summary(fitobj, prob = c(0.025, 0.975))$summary
    if(algorithm == "approx"){
      outsel <- summ[grep("theta\\[", rownames(summ)), c("mean", "2.5%", "97.5%")]
    } else if(algorithm == "exact"){
      outsel <- summ[grep("beta\\[", rownames(summ)), c("mean", "2.5%", "97.5%")]
    }
    
    res <- cbind.data.frame(nms, prior, algorithm, outsel)
    return(res)
  }
  
  res.ridge1 <- get.results(fitobj = fit.ridge, prior = "ridge", algorithm = "exact")
  res.lasso1 <- get.results(fitobj = fit.lasso, prior = "lasso", algorithm = "exact")
  res.hs1 <- get.results(fitobj = fit.hs, prior = "hs", algorithm = "exact")
  
  res.exact <- rbind.data.frame(res.ridge1, res.lasso1, res.hs1)
  
  ## add results shrinkem
  res.ridge <- summary(shrink.ridge)
  res.ridge <- res.ridge[, c(grep("shrunk.mean|shrunk.lower|shrunk.upper", colnames(res.ridge)))]
  colnames(res.ridge) <- c("Mean", "LB", "UB")
  res.ridge$Variable <- rownames(res.ridge)
  res.ridge$Prior <- "ridge"
  res.ridge$Algorithm <- "shrinkem"
  
  res.lasso <- summary(shrink.lasso)
  res.lasso <- res.lasso[, c(grep("shrunk.mean|shrunk.lower|shrunk.upper", colnames(res.lasso)))]
  colnames(res.lasso) <- c("Mean", "LB", "UB")
  res.lasso$Variable <- rownames(res.lasso)
  res.lasso$Prior <- "lasso"
  res.lasso$Algorithm <- "shrinkem"
  
  res.hs <- summary(shrink.hs)
  res.hs <- res.hs[, c(grep("shrunk.mean|shrunk.lower|shrunk.upper", colnames(res.hs)))]
  colnames(res.hs) <- c("Mean", "LB", "UB")
  res.hs$Variable <- rownames(res.hs)
  res.hs$Prior <- "hs"
  res.hs$Algorithm <- "shrinkem"
  
  res.shrink <- rbind.data.frame(res.ridge, res.lasso, res.hs)
  colnames(res.exact) <- c("Variable", "Prior", "Algorithm", "Mean", "LB", "UB")
  res <- rbind.data.frame(res.exact, res.shrink)
  
  # Compute PMSE
  testX <- as.data.frame(t(test[, -grep("ViolentCrimesPerPop", colnames(test))]))
  testX$Variable <- rownames(testX)
  testY <- test$ViolentCrimesPerPop
  
  res$Method <- factor(paste(res$Prior, res$Algorithm, sep = "_"))
  out[[k]] <- data.frame(NA)
  for(i in 1:length(levels(res$Method))){
    sel <- res[which(res$Method == levels(res$Method)[i]), c("Variable", "Mean")]
    comb <- merge(sel, testX, by = "Variable")
    
    test.obs <- comb[, -c(grep("Variable|Mean", colnames(comb)))]
    est <- comb$Mean
    predY <- apply(test.obs, 2, function(x) sum(est*x))
    pmse <- mean((testY - predY)^2)
    
    out[[k]][i, 1] <- levels(res$Method)[i]
    out[[k]][i, 2] <- pmse
  }
  
  colnames(out[[k]]) <- c("Method", "PMSE")
  
  # add MSE regular lm
  sel <- data.frame("Estimate" = lmfit$coefficients,
                    "Variable" = names(lmfit$coefficients))
  comb <- merge(sel, testX, by = "Variable")
  test.obs <- comb[, -c(grep("Variable|Estimate", colnames(comb)))]
  est <- comb$Estimate
  predY <- apply(test.obs, 2, function(x) sum(est*x))
  pmse <- mean((testY - predY)^2)
  out[[k]][7, 1] <- "lm"
  out[[k]][7, 2] <- pmse
}

save(out, file ="./results/CV_PMSE_crime_half.RData")

load("./results/CV_PMSE_crime_half.RData")

pmse <- do.call(rbind.data.frame, out)
pmse$Method <- plyr::revalue(pmse$Method, 
                             c("hs_exact" = "Exact horseshoe",
                               "hs_shrinkem" = "App. horseshoe",
                               "lasso_exact" = "Exact lasso ",
                               "lasso_shrinkem" = "App. lasso",
                               "ridge_exact" = "Exact ridge",
                               "ridge_shrinkem" = "App. ridge",
                               "lm" = "Unregularized"))

sel <- pmse[which(pmse$Method != "Unregularized"), ]

png(file = "./results/CV_PMSE_crime_half_reg.png", width = 1000, height = 800)
ggplot(sel, aes(x = Method, y = PMSE)) +
  geom_boxplot() +
  scale_x_discrete(guide = guide_axis(angle = 90)) +
  theme_bw(base_size = 25)
dev.off()

sel <- pmse[which(pmse$Method == "Unregularized"), ]

png(file = "./results/CV_PMSE_crime_half_unreg.png", width = 1000, height = 800)
ggplot(sel, aes(x = Method, y = PMSE)) +
  geom_boxplot() +
  scale_x_discrete(guide = guide_axis(angle = 90)) +
  theme_bw(base_size = 25)
dev.off()

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
mod.mat <- model.matrix(~., df)[, -1] # removes all NAs

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
save(res, file = "./results/full_results_df.RData")

# Visualize estimates
load("./results/full_results_df.RData")

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
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) + 
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom", legend.key.width = unit(1.5, "cm"))
dev.off()

##### Results: PMSE ------
## PMSE is computed based on unselected estimates
## 95% interval is not ideal to select predictors, so I expect this to worsen the PMSE

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



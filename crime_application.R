# Crime application using approximate Bayesian regularization
## Author: Sara van Erp

library(rstan)
options(mc.cores = 4)
rstan_options(auto_write = TRUE)
library(shrinkem)
library(brms)

set.seed(07042023)

##### Data ------
# Data can be downloaded from: https://archive.ics.uci.edu/ml/datasets/communities+and+crime+unnormalized
dat <- read.table("./communities_crime_unnormalized.txt", sep=",", na.string="?")

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


# 1 predictor is treated as integer, but is really nominal 
class(dat$LemasGangUnitDeploy)
dat$LemasGangUnitDeploy <- factor(dat$LemasGangUnitDeploy, levels=c(0, 5, 10), labels=c("no", "parttime", "yes"))

# remove non-predictive attributes and state 
# state is included in the original application, but excluded here to avoid having p > n
dat.sel <- subset(dat, select=-c(state, communityname, countyCode, communityCode, fold))

# remove possible prediction goals, keep only the total number of violent crimes to predict (others are subtotals) 
df <- subset(dat.sel, select=-c(murders, murdPerPop, rapes, rapesPerPop, robberies, robbPerPop, assaults, assaultPerPop, burglaries,
                                burglPerPop, larcenies, larcPerPop, autoTheft, autoTheftPerPop, arsons, arsonsPerPop, nonViolPerPop))

summary(df)

# plot outcome measure 
hist(df$ViolentCrimesPerPop) # skewed

# log transform the outcome measure
hist(log(df$ViolentCrimesPerPop)) # more normal
df$ViolentCrimesPerPop <- log(df$ViolentCrimesPerPop)

# create design matrix
mod.mat <- model.matrix(~., df) # removes all NAs

# for simplicity: keep only the continuous predictors. 
# The original application included dummies for nominal predictors as well, but since these cannot be standardized, their coefficients might be influenced differently by the shrinkage priors
# This issue needs additional checking
# OwnOccQrange and RentQrange were removed too since these are functions of other predictors leading to singularities in the MLEs
cont.mat <- mod.mat[, -c(grep("(Intercept)|LemasGangUnitDeployparttime|LemasGangUnitDeployyes|OwnOccQrange|RentQrange", colnames(mod.mat)))]

# 90% training and 10% test set and standardize both
ntrain <- as.integer(0.9*nrow(cont.mat))
ntest <- nrow(cont.mat)-ntrain

train <- data.frame(scale(cont.mat[1:ntrain, ]))
test <- data.frame(scale(cont.mat[(ntrain+1):nrow(cont.mat), ]))

##### Get maximum likelihood estimates -----
lmfit <- lm(train$ViolentCrimesPerPop ~ -1 + ., train)
summary(lmfit)

## extract MLEs
# note: first entry corresponds to the intercept
mle <- coef(lmfit)
covmat <- vcov(lmfit)

##### Fit approximate regularization model using Stan -----
ABR_fun <- function(mle, covmat, prior = c("ridge", "lasso", "hs"), 
                    iter = 4000, s0 = 1, nu0 = 3){ 
  if(prior  == "ridge"){
    standat <- list(p = length(mle),
                    mle = mle,
                    errorcov = covmat,
                    s0 = s0,
                    nu0 = nu0)
    mod <- stan_model("./models/approx_regression_ridge.stan")
  }
  
  if(prior == "lasso"){
    standat <- list(p = length(mle),
                    mle = mle,
                    errorcov = covmat,
                    s0 = s0,
                    nu0 = nu0)
    mod <- stan_model("./models/approx_regression_lasso.stan")
  }
  
  if(prior == "lassoNS"){
    standat <- list(p = length(mle),
                    mle = mle,
                    errorcov = covmat,
                    s0 = s0,
                    nu0 = nu0)
    mod <- stan_model("./models/approx_regression_lasso_notScaled.stan")
  }
  
  if(prior == "hs"){
    standat <- list(p = length(mle),
                    mle = mle,
                    errorcov = covmat,
                    s0 = s0)
    mod <- stan_model("./models/approx_regression_hs.stan")
  }
  
  fit <- sampling(mod, data = standat, iter = iter)
  save(fit, file = paste0("./results/fit_approx_", prior, "_crime.RData"))
}

prior <- "ridge"
fit <- ABR_fun(mle = mle, covmat = covmat, prior = prior)
prior <- "lasso"
fit <- ABR_fun(mle = mle, covmat = covmat, prior = prior) # large Rhat
prior <- "lassoNS"
fit <- ABR_fun(mle = mle, covmat = covmat, prior = prior) # does appear to converge
prior <- "hs"
fit <- ABR_fun(mle = mle, covmat = covmat, prior = prior) # gives divergences

##### Compare with exact solution -----
ex_fun <- function(train, prior = c("ridge", "lasso", "hs"), 
                    iter = 4000, s0 = 1, nu0 = 3){ 
  
  input.dat <- list(N_train = nrow(train),
                    p = ncol(train)-1,
                    y_train = train$ViolentCrimesPerPop,
                    X_train = train[, -c(grep("ViolentCrimesPerPop", colnames(train)))]
  )
  
  if(prior  == "ridge"){
    standat <- c(input.dat, 
                 list(s0 = s0,
                    nu0 = nu0))
    mod <- stan_model("./models/exact_regression_ridge.stan")
  }
  
  if(prior == "lasso"){
    standat <- c(input.dat, 
                 list(s0 = s0,
                      nu0 = nu0))
    mod <- stan_model("./models/exact_regression_lasso.stan")
  }
  
  if(prior == "hs"){
    standat <- c(input.dat, 
                 list(s0 = s0))
    mod <- stan_model("./models/exact_regression_hs.stan")
  }
  
  fit <- sampling(mod, data = standat, iter = iter)
  save(fit, file = paste0("./results/fit_exact_", prior, "_crime.RData"))
}

prior <- "ridge"
fit <- ex_fun(train = train, prior = prior)
prior <- "lasso"
fit <- ex_fun(train = train, prior = prior)
prior <- "hs"
fit <- ex_fun(train = train, prior = prior) # gives divergences

##### Combine results -----
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

load("./results/fitobjects/fit_approx_ridge_crime.RData")
res.ridge1 <- get.results(fitobj = fit, prior = "ridge", algorithm = "approx")
load("./results/fitobjects/fit_approx_lasso_crime.RData")
res.lasso1 <- get.results(fitobj = fit, prior = "lasso", algorithm = "approx")
load("./results/fitobjects/fit_approx_lassoNS_crime.RData")
res.lassoNS <- get.results(fitobj = fit, prior = "lassoNS", algorithm = "approx")
load("./results/fitobjects/fit_approx_hs_crime.RData")
res.hs1 <- get.results(fitobj = fit, prior = "hs", algorithm = "approx")

load("./results/fitobjects/fit_exact_ridge_crime.RData")
res.ridge2 <- get.results(fitobj = fit, prior = "ridge", algorithm = "exact")
load("./results/fitobjects/fit_exact_lasso_crime.RData")
res.lasso2 <- get.results(fitobj = fit, prior = "lasso", algorithm = "exact")
load("./results/fitobjects/fit_exact_hs_crime.RData")
res.hs2 <- get.results(fitobj = fit, prior = "hs", algorithm = "exact")

res <- rbind.data.frame(res.ridge1, res.lasso1, res.hs1, res.lassoNS,
                        res.ridge2, res.lasso2, res.hs2)

# Select variables based on 95% CI 
sel <- rep(NA, nrow(res))
for(i in 1:nrow(res)){
  sel[i] <- ifelse(res$`2.5%`[i] <= 0 & res$`97.5%`[i] >=0, FALSE, TRUE)
}

res$select <- sel

sum(res$select)

##### Compare with shrinkem package -----
shrink.ridge <- shrinkem(mle, Sigma = covmat, type = "ridge")
save(shrink.ridge, file = "./results/fit_shrinkem_ridge_crime.RData")
shrink.lasso <- shrinkem(mle, Sigma = covmat, type = "lasso")
save(shrink.lasso, file = "./results/fit_shrinkem_lasso_crime.RData")
shrink.hs <- shrinkem(mle, Sigma = covmat, type = "horseshoe") # doesn't run

##### Compare with brms -----
# note: the ridge is different because the sd is not estimated based on data and the horseshoe in brms is a regularized horseshoe instead of a normal one
# fixed the sd_ridge to estimated value from exact implementation
brms.ridge <-  brm(ViolentCrimesPerPop ~ -1 + ., data = train, prior = prior(normal(0, 0.05), class = "b"))
save(brms.ridge, file = "./results/fit_brms_ridge_crime.RData")
brms.lasso <- brm(ViolentCrimesPerPop ~ -1 + ., data = train, prior = prior(lasso(), class = "b"))
save(brms.lasso, file = "./results/fit_brms_lasso_crime.RData")
brms.hs <- brm(ViolentCrimesPerPop ~ -1 + ., data = train, prior = prior(horseshoe(1), class = "b")) # less divergences than actual horseshoe
save(brms.hs, file = "./results/fit_brms_hs_crime.RData")

##### Combine all results in df -----
load("./results/fitobjects/fit_shrinkem_ridge_crime.RData")
load("./results/fitobjects/fit_shrinkem_lasso_crime.RData")

load("./results/fitobjects/fit_brms_ridge_crime.RData")
load("./results/fitobjects/fit_brms_lasso_crime.RData")
load("./results/fitobjects/fit_brms_hs_crime.RData")

## exact and approximate Stan results
head(res)
colnames(res) <- c("Variable", "Prior", "Algorithm", "Estimate", "LB", "UB", "Included")

## brms
res.ridge <- data.frame(fixef(brms.ridge)[, -2])
colnames(res.ridge) <- c("Estimate", "LB", "UB")
res.ridge$Variable <- rownames(res.ridge)
res.ridge$Prior <- "ridge"
res.ridge$Algorithm <- "brms"

res.lasso <- data.frame(fixef(brms.lasso)[, -2])
colnames(res.lasso) <- c("Estimate", "LB", "UB")
res.lasso$Variable <- rownames(res.lasso)
res.lasso$Prior <- "lasso"
res.lasso$Algorithm <- "brms"

res.hs <- data.frame(fixef(brms.hs)[, -2])
colnames(res.hs) <- c("Estimate", "LB", "UB")
res.hs$Variable <- rownames(res.hs)
res.hs$Prior <- "hs"
res.hs$Algorithm <- "brms"

res.brms <- rbind.data.frame(res.ridge, res.lasso, res.hs)

# Select variables based on 95% CI 
sel <- rep(NA, nrow(res.brms))
for(i in 1:nrow(res.brms)){
  sel[i] <- ifelse(res.brms$LB[i] <= 0 & res.brms$UB[i] >=0, FALSE, TRUE)
}

res.brms$Included <- sel

## shrinkem
res.ridge <- summary(shrink.ridge)
res.ridge <- res.ridge[, c(grep("shrunk.mean|shrunk.lower|shrunk.upper|nonzero", colnames(res.ridge)))]
colnames(res.ridge) <- c("Estimate", "LB", "UB", "Included")
res.ridge$Variable <- rownames(res.ridge)
res.ridge$Prior <- "ridge"
res.ridge$Algorithm <- "shrinkem"

res.lasso <- summary(shrink.lasso)
res.lasso <- res.lasso[, c(grep("shrunk.mean|shrunk.lower|shrunk.upper|nonzero", colnames(res.lasso)))]
colnames(res.lasso) <- c("Estimate", "LB", "UB", "Included")
res.lasso$Variable <- rownames(res.lasso)
res.lasso$Prior <- "lasso"
res.lasso$Algorithm <- "shrinkem"

res.shrink <- rbind.data.frame(res.ridge, res.lasso)

res <- rbind.data.frame(res, res.brms, res.shrink)
save(res, file = "./results/full_results_df.RData")

##### Plot results -----
load("./results/full_results_df.RData")

pd <- position_dodge(0.8)
# reorder predictors based on mle value
ord <- names(sort(abs(mle), decreasing = TRUE))
res$Variable <- factor(res$Variable, levels = ord)

# ridge
df.sel <- res[which(res$Prior == "ridge" & res$Variable %in% ord[1:21]), ]
png(file = "./results/crime_est_ridge1.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) + 
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

df.sel <- res[which(res$Prior == "ridge" & res$Variable %in% ord[22:71]), ]
png(file = "./results/crime_est_ridge2.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) + 
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

df.sel <- res[which(res$Prior == "ridge" & res$Variable %in% ord[72:121]), ]
png(file = "./results/crime_est_ridge3.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) + 
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

# lasso
# change algorithm to distinguish between scaled and unscaled lasso
for(i in 1:nrow(res)){
  if(res$Prior[i] == "lassoNS"){
    res$Algorithm[i] <- "approx_notScaled"
  }
}
df.sel <- res[which(res$Prior %in% c("lasso", "lassoNS") & res$Variable %in% ord[1:4]), ]

png(file = "./results/crime_est_lasso0.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) +
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

df.sel <- res[which(res$Prior %in% c("lasso", "lassoNS") & res$Variable %in% ord[5:21]), ]
png(file = "./results/crime_est_lasso1.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) +
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

df.sel <- res[which(res$Prior == "lasso" & res$Variable %in% ord[22:71]), ]
png(file = "./results/crime_est_lasso2.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) +
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

df.sel <- res[which(res$Prior == "lasso" & res$Variable %in% ord[72:121]), ]
png(file = "./results/crime_est_lasso3.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) + 
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

# horseshoe
df.sel <- res[which(res$Prior == "hs" & res$Variable %in% ord[1:21]), ]
png(file = "./results/crime_est_hs1.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) + 
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

df.sel <- res[which(res$Prior == "hs" & res$Variable %in% ord[22:71]), ]
png(file = "./results/crime_est_hs2.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) + 
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

df.sel <- res[which(res$Prior == "hs" & res$Variable %in% ord[72:121]), ]
png(file = "./results/crime_est_hs3.png", width = 1000, height = 1300)
ggplot(df.sel, aes(x = Estimate, y = Variable, colour = Algorithm)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 1.3) +
  ylab("Variable") + xlab("Posterior mean and 95% CI") + theme_bw(base_size = 25) + 
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom")
dev.off()

##### PMSE -----
## PMSE is computed based on unselected estimates
## 95% interval is not ideal to select predictors, so I expect this to worsen the PMSE

testX <- as.data.frame(t(test[, -grep("ViolentCrimesPerPop", colnames(test))]))
testX$Variable <- rownames(testX)
testY <- test$ViolentCrimesPerPop

res$Method <- factor(paste(res$Prior, res$Algorithm, sep = "_"))
out <- data.frame(NA)
for(i in 1:length(levels(res$Method))){
  sel <- res[which(res$Method == levels(res$Method)[i]), c("Variable", "Estimate")]
  comb <- merge(sel, testX, by = "Variable")
  
  test.obs <- comb[, -c(grep("Variable|Estimate", colnames(comb)))]
  est <- comb$Estimate
  predY <- apply(test.obs, 2, function(x) sum(est*x))
  pmse <- mean((testY - predY)^2)
  
  out[i, 1] <- levels(res$Method)[i]
  out[i, 2] <- pmse
}

colnames(out) <- c("Method", "PMSE")
out

##### Variational Bayes in Stan -----
# An interesting comparison would be with the vb algorithm in Stan
# This also provides an approximate (and fast) method, but based on the exact model implementation
standat <-  list(N_train = nrow(train),
               p = ncol(train)-1,
               y_train = train$ViolentCrimesPerPop,
               X_train = train[, -c(grep("ViolentCrimesPerPop", colnames(train)))],
               s0 = 1,
               nu0 = 3)
mod <- stan_model("./models/exact_regression_lasso.stan")
fit.vb <- vb(mod, data = standat)
summary(fit.vb)$summary


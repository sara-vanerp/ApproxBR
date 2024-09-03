# Measurement invariance application using approximate Bayesian regularization
## Author: Sara van Erp

library(psychTools) # for the complete HolzingerSwineford1939 data
library(lavaan)
library(regsem)
library(shrinkem)
library(blavaan)
library(ggplot2)
library(dplyr)

seed <- 29092023
set.seed(seed)

##### Data preparation ------
# Based on: Liang & Jacobucci (2020): Regularized Structural Equation Modeling 
# to Detect Measurement Bias: Evaluation of Lasso, Adaptive Lasso, and Elastic Net
# Note: LiangJacobucci2020 use regsem and regularize all cross-loadings as well, 
# but that model is not identified classically, so this is not possible here

# Recode age as in LiangJacobucci2020
holzinger.swineford$age <- holzinger.swineford$ageyr + holzinger.swineford$mo/12

# 90% training and 10% test set and standardize both
ntrain <- as.integer(0.9*nrow(holzinger.swineford))
ntest <- nrow(holzinger.swineford)-ntrain

sel <- grep("t|age", colnames(holzinger.swineford))
train <- data.frame(scale(holzinger.swineford[1:ntrain, sel]))
test <- data.frame(scale(holzinger.swineford[(ntrain+1):nrow(holzinger.swineford), sel]))

##### Analyses ------

# Option 1: Approximate with shrinkem (ridge and horseshoe priors)

# run the model with lavaan
# note: LiangJacobucci2020 use this model to detect uniform bias only, which is why the interaction terms eta*age are not included
# LiangJacobucci2020 describe using t26 instead of t04 but due to missing data in t26, t04 is used here.
mod_lav <- '
  spatial =~ t01_visperc + t02_cubes + t03_frmbord + t04_lozenges
  verbal =~ t05_geninfo + t06_paracomp + t07_sentcomp + t08_wordclas + t09_wordmean
  speed =~ t10_addition + t11_code + t12_countdot + t13_sccaps
  memory =~ t14_wordrecg + t15_numbrecg + t16_figrrecg + t17_objnumb + t18_numbfig + t19_figword
  t01_visperc ~ age
  t02_cubes ~ age
  t03_frmbord ~ age
  t04_lozenges ~ age
  t05_geninfo ~ age
  t06_paracomp ~ age
  t07_sentcomp ~ age
  t08_wordclas ~ age
  t09_wordmean ~ age
  t10_addition ~ age
  t11_code ~ age
  t12_countdot ~ age
  t13_sccaps ~ age
  t14_wordrecg ~ age
  t15_numbrecg ~ age
  t16_figrrecg ~ age
  t17_objnumb ~ age
  t18_numbfig ~ age
  t19_figword ~ age
'

fit_lav <- sem(mod_lav, data = train, std.lv = TRUE)

# extract MLEs and covariance matrix for regression parameters corresponding to age
mle <- coef(fit_lav)
mleSel <- mle[grep("~age", names(mle))]
covmat <- lavInspect(fit_lav, what = "vcov")
id <- grep("~age", colnames(covmat))
covmatSel <- covmat[id, id]

# shrinkem with ridge prior
shrink_ridge <- shrinkem(mleSel, covmatSel, type="ridge")
save(shrink_ridge, file = "./results/fitobjects/fit_shrinkem_ridge_MI.RData")

# shrinkem with horseshoe prior
shrink_hs <- shrinkem(mleSel, covmatSel, type="horseshoe")
save(shrink_hs, file = "./results/fitobjects/fit_shrinkem_hs_MI.RData")

# extract estimated prior variance for use in blavaan
d <- shrink_ridge$draws
lam2 <- d$lambda2
lam <- sqrt(lam2)
mean(lam) # 0.1

# Option 2: Exact with blavaan (ridge prior)
ridgePriors <- dpriors(beta = "normal(0, 0.1)") # use the estimated prior sd from shrinkem

fit_blav <- bsem(mod_lav, 
                 data = train, 
                 std.lv = TRUE, 
                 dp = ridgePriors,
                 bcontrol = list(cores = 3),
                 seed = seed)
save(fit_blav, file = "./results/fitobjects/fit_blavaan_ridge_MI.RData")

# option 3: Classical with regsem (ridge and elastic net penalties)
# same settings as in LiangJacobucci2020, only AIC no longer seems possible as metric

# ridge
cv_fit_ridge <- cv_regsem(fit_lav,
                    pars_pen = "regressions",
                    metric = "BIC",
                    n.lambda = 15,
                    jump = 0.05,
                    lambda.start = 0,
                    alpha = 1,
                    type = "ridge")

save(cv_fit_ridge, file = "./results/fitobjects/fit_regsem_ridge_MI.RData")

# elastic net
cv_fit_en <- cv_regsem(fit_lav,
                    pars_pen = "regressions",
                    metric = "BIC",
                    n.lambda = 15,
                    jump = 0.05,
                    lambda.start = 0,
                    alpha = 0.5,
                    type = "enet")

save(cv_fit_en, file = "./results/fitobjects/fit_regsem_enet_MI.RData")

##### Results: Estimation ------

# shrinkem
load("./results/fitobjects/fit_shrinkem_ridge_MI.RData")
res1 <- cbind.data.frame(rownames(shrink_ridge$estimates),
                         shrink_ridge$estimates[, c('shrunk.mean', 'shrunk.mode', 'shrunk.lower', 'shrunk.upper')],
                         "shrinkem", "ridge")
colnames(res1) <- c("par", "mean", "mode", "ci.lower", "ci.upper", "package", "prior")

load("./results/fitobjects/fit_shrinkem_hs_MI.RData")
res2 <- cbind.data.frame(rownames(shrink_hs$estimates),
                         shrink_hs$estimates[, c('shrunk.mean', 'shrunk.mode', 'shrunk.lower', 'shrunk.upper')],
                         "shrinkem", "hs")
colnames(res2) <- c("par", "mean", "mode", "ci.lower", "ci.upper", "package", "prior")

# blavaan
load("./results/fitobjects/fit_blavaan_ridge_MI.RData")
out_blav <- parameterEstimates(fit_blav)
out_blav$par <- paste(out_blav$lhs, out_blav$op, out_blav$rhs, sep = "")
ci_blav <- blavInspect(fit_blav, what = "hpd", level = 0.95)
mode_blav <- blavInspect(fit_blav, what = "postmode")
out_blav_sel <- out_blav[which(out_blav$par %in% rownames(ci_blav)), ]
res3 <- cbind.data.frame(out_blav_sel[, c('est', 'par')], ci_blav, mode_blav)
colnames(res3) <- c("mean", "par", "ci.lower", "ci.upper", "mode")
res3$package <- "blavaan"
res3$prior <- "ridge"

# regsem
load("./results/fitobjects/fit_regsem_ridge_MI.RData")
est_regsem_ridge <- cv_fit_ridge$final_pars
res4 <- cbind.data.frame(names(est_regsem_ridge), est_regsem_ridge, NA, NA, "regsem", "ridge", NA)
colnames(res4) <- c("par", "mean", "ci.lower", "ci.upper", "package", "prior", "mode")

load("./results/fitobjects/fit_regsem_enet_MI.RData")
est_regsem_en <- cv_fit_en$final_pars
res5 <- cbind.data.frame(names(est_regsem_en), est_regsem_en, NA, NA, "regsem", "enet", NA)
colnames(res5) <- c("par", "mean", "ci.lower", "ci.upper", "package", "prior", "mode")

# combine
res <- rbind.data.frame(res1, res2, res3, res4, res5)
# change naming effects of age to be in line across packages
parsel <- grep("age -> ", res$par)
res$par[parsel] <- paste0(gsub("age -> ", "", res$par[parsel]), "~age")

# select variables based on 95% CI 
# 1 if relevant and selected, 0 if not
res$sel <- ifelse(res$ci.lower > 0 | res$ci.upper < 0, 1, 0)
# add selection regsem
sel <- which(res$package == "regsem")
sel.regsem <- ifelse(res[sel, "mean"] == 0, 0, 1)
res$sel[sel] <- sel.regsem

res$meth <- paste(res$prior, res$package, sep = " ")

save(res, file = "./results/full_results_MI.RData")

res.sel <- res[grep("age", res$par), ]
res.sel %>%
  group_by(meth) %>%
  summarise(sel = sum(sel))

# Visualize estimates
load("./results/full_results_MI.RData")
colnames(res) <- c("Variable", "Mean", "Mode", "LB", "UB", "Package", "Prior", "Included", "Method")

# select only regularized parameters
sel <- res[grep("age", res$Variable), ]

# order based on mean for shrinkem ridge
ord <- res1[order(abs(res1$mean), decreasing = TRUE), "par"]
sel$Variable <- factor(sel$Variable, levels = ord)

# change names factor levels
sel$Method <- factor(sel$Method)
levels(sel$Method) <- list("Elastic net regsem" = "enet regsem",
                           "Horseshoe shrinkem" = "hs shrinkem",
                           "Ridge blavaan" = "ridge blavaan",
                           "Ridge regsem" = "ridge regsem",
                           "Ridge shrinkem" = "ridge shrinkem")

# decided to remove the elastic net from the results because it pulls all effects to zero
sel2 <- sel %>%
  filter(Method != "Elastic net regsem") %>%
  droplevels()

pd <- position_dodge(0.8)

png(file = "./results/MI_comparison_priors.png", width = 1000, height = 1300)
ggplot(sel2, aes(x = Mean, y = Variable, colour = Method, linetype = Method)) +
  geom_errorbar(aes(xmin = LB, xmax = UB), position = pd, linewidth = 1) +
  geom_point(position = pd, size = 3) +
  geom_point(aes(x = Mode), position = pd, size = 3, shape = 17) +
  scale_linetype_manual("", values = c(1, 2, 3, 1)) +
  scale_colour_manual("", values = c("black", "blue", "blue", "blue")) + 
  ylab("Variable") + xlab("Posterior estimates and 95% CI") + theme_bw(base_size = 25) + 
  guides(colour = guide_legend(nrow = 2), linetype = guide_legend(nrow = 2)) +
  theme(axis.text.x = element_text(angle = 90), legend.title = element_blank(), legend.position = "bottom", legend.key.width = unit(1.5, "cm")) 
dev.off()

##### Results: PMSE ------
# for a simplified PMSE, ignore the influence of the latent variable since this is constant across methods
testY <- test[, 3:21]
pmse_fun <- function(testY, predY){
  pmse = sum(colSums((testY-predY)^2))/(ncol(testY)*nrow(testY))
  return(pmse)
}

# unregularized solution lavaan
gamma <- parameterestimates(fit_lav)[20:38, "est"]

predY <- sapply(gamma, function(x, xtest = test$age){
  x*xtest
})

# PMSE
pmse_fun(testY, predY)

# shrinkage methods
out <- data.frame(NA)
for(i in 1:length(levels(sel$Method))){
  gamma <- sel[which(sel$Method == levels(sel$Method)[i]), "Mean"]
  
  predY <- sapply(gamma, function(x, xtest = test$age){
    x*xtest
  })
  
  pmse <- pmse_fun(testY, predY)
  
  out[i, 1] <- levels(sel$Method)[i]
  out[i, 2] <- pmse
}

colnames(out) <- c("Method", "PMSE")
print(out, digits = 2)

# split PMSE per variable
pmse_var <- function(testY, predY){
  pmse <- rep(NA, ncol(testY))
  for(p in 1:ncol(testY)){
    pmse[p] <- mean((testY[, p]-predY[, p])^2)
  }
  return(pmse)
}

# unregularized solution lavaan
gamma <- parameterestimates(fit_lav)[20:38, "est"]

predY <- sapply(gamma, function(x, xtest = test$age){
  x*xtest
})

# PMSE
pmse_var_lav <- pmse_var(testY, predY)

# shrinkage methods
out <- data.frame(NA)
for(i in 1:length(levels(sel$Method))){
  gamma <- sel[which(sel$Method == levels(sel$Method)[i]), "Mean"]
  
  predY <- sapply(gamma, function(x, xtest = test$age){
    x*xtest
  })
  
  pmse <- pmse_var(testY, predY)
  
  out[i, 1] <- levels(sel$Method)[i]
  out[i, 2:20] <- pmse
}

out
out[6, 1] <- "lavaan"
out[6, 2:20] <- pmse_var_lav
print(out, digits = 2)



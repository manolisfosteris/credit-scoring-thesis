# =============================================================================
# Credit Scoring Evaluation Using Machine Learning
# Diploma Thesis - National Technical University of Athens (NTUA)
# Author: Emmanouil Fosteris
# Supervisor: Prof. Chryseis Karoni
# September 2022
# =============================================================================

# --- Libraries ---------------------------------------------------------------
library(dplyr)
library(pROC)
library(glmnet)
library(ggplot2)
library(rpart)
library(cowplot)
library(randomForest)
library(tree)
library(xtable)
library(glmtoolbox)
library(lmtest)
library(rpart.plot)

# --- Data Import -------------------------------------------------------------

# Training data
training_data <- read.table(choose.files(), header = TRUE)

# Remove ID column
training_data <- training_data[, -1]

# Convert columns to factors
for (i in 2:6) {
  training_data[, i] <- factor(training_data[, i])
}

levels(training_data$athetisi) <- c('No', 'Yes')
levels(training_data$solvency) <- c('0', '1')
levels(training_data$property) <- c('0', '1', '2')
levels(training_data$history)  <- c('0', '1', '2')
levels(training_data$employment) <- c('0', '1', '2', '3')

attach(training_data)

# Test data
test.data <- read.table(choose.files(), header = TRUE)
test.data <- select(test.data, -1)

for (i in 1:5) {
  test.data[, i] <- factor(test.data[, i])
}

levels(test_data$solvency) <- c('0', '1')
levels(test_data$property) <- c('0', '1', '2')
levels(test_data$history)  <- c('0', '1', '2')
levels(test_data$employment) <- c('0', '1', '2', '3')

# --- Contingency Tables & Chi-squared Tests ----------------------------------

freq.table1 <- table(athetisi, solvency)
freq.table2 <- table(athetisi, property)
freq.table3 <- table(athetisi, history)
freq.table4 <- table(athetisi, employment)

chisq1 <- chisq.test(freq.table1)
chisq2 <- chisq.test(freq.table2)
chisq3 <- chisq.test(freq.table3)
chisq4 <- chisq.test(freq.table4)

# --- Logistic Regression (Full Model) ----------------------------------------

glm.fits <- glm(athetisi ~ ., data = training_data, family = binomial)
summary(glm.fits)
coef(glm.fits)
summary(glm.fits)$coef

# --- Logistic Regression (Without History) -----------------------------------

glm.fits2 <- glm(athetisi ~ solvency + property + employment,
                  data = training_data, family = binomial)
summary(glm.fits2)

# --- Residual Plots ----------------------------------------------------------

# Pearson residuals
pearson.res <- residuals(glm.fits2, type = "pearson")
st.pearson.res <- residuals(glm.fits2, type = "pearson") /
  (sqrt(1 - hatvalues(glm.fits2)))

par(mfrow = c(1, 2))
plot(fitted.values(glm.fits2), st.pearson.res,
     xlab = "Fitted values", ylab = "Standard pearson residuals")
abline(h = 0)
plot(id, st.pearson.res, ylab = "Standard pearson residuals")
abline(h = 0)

# Deviance residuals
deviance.res <- residuals(glm.fits2, type = "deviance")
st.deviance.res <- residuals(glm.fits2, type = "deviance") /
  (sqrt(1 - hatvalues(glm.fits2)))

par(mfrow = c(1, 2))
plot(fitted.values(glm.fits2), st.deviance.res,
     xlab = "Fitted values", ylab = "Standard deviance residuals")
abline(h = 0)
plot(id, st.deviance.res, ylab = "Standard deviance residuals")
abline(h = 0)

# Likelihood residuals
y <- as.numeric(training_data$athetisi) - 1
res.lik <- sign(y - fitted.values(glm.fits2)) *
  sqrt((hatvalues(glm.fits2) * (st.pearson.res)^2) +
       ((1 - hatvalues(glm.fits2)) * (st.deviance.res)^2))

par(mfrow = c(1, 2))
plot(hatvalues(glm.fits2), res.lik,
     xlab = "Fitted values", ylab = "Likelihood residuals")
abline(h = 0)
plot(training_data$id, res.lik, ylab = "Likelihood residuals")
abline(h = 0)

# Cook's distance & Hat values
par(mfrow = c(1, 2))
plot(training_data$id, cooks.distance(glm.fits2),
     ylab = "Cook's distance", xlab = 'id')
plot(training_data$id, hatvalues(glm.fits2),
     ylab = "Hat values", xlab = 'id')

# --- ROC Curve (Training Set) ------------------------------------------------

par(mfrow = c(1, 1))
roc(athetisi, fitted.values(glm.fits2),
    ci = TRUE, smooth = TRUE, plot = TRUE,
    xlab = '1-Specificity',
    main = 'ROC curve for training set')

# --- ROC Curve (Test Set) ----------------------------------------------------

glm.probs <- predict(glm.fits2, newdata = test_data, type = "response")
roc(test_data$athetisi, glm.probs,
    ci = TRUE, smooth = TRUE, plot = TRUE,
    xlab = '1-Specificity',
    main = 'ROC curve for test set')

# --- Ridge Regression --------------------------------------------------------

x.train <- model.matrix(athetisi ~ ., training_data)[, -1]
y.train <- training_data$athetisi

cv.ridge <- cv.glmnet(x.train, y.train, alpha = 0, family = "binomial")

ridge.model <- glmnet(x.train, y.train, alpha = 0, family = "binomial",
                       lambda = cv.ridge$lambda.min)

x.test <- model.matrix(athetisi ~ ., test_data)[, -1]
y.test <- test_data$athetisi

ridge.probs <- predict(ridge.model, s = cv.ridge$lambda.min,
                        type = "response", newx = x.test)
ridge.pred <- ifelse(ridge.probs > 0.5, "1", "0")

cat("Optimal lambda (Ridge):", cv.ridge$lambda.min, "\n")
cat("Accuracy (Ridge):", mean(ridge.pred == y.test), "\n")
table(ridge.pred, test_data$athetisi)

# ROC for Ridge
roc(test_data$athetisi, as.numeric(ridge.probs),
    ci = TRUE, smooth = TRUE, plot = TRUE,
    xlab = '1-Specificity',
    main = 'ROC curve for ridge regression')

# --- Lasso Regression --------------------------------------------------------

cv.lasso <- cv.glmnet(x.train, y.train, alpha = 1, family = "binomial")

lasso.model <- glmnet(x.train, y.train, alpha = 1, family = "binomial",
                       lambda = cv.lasso$lambda.min)

lasso.probs <- predict(lasso.model, s = cv.lasso$lambda.min,
                        type = "response", newx = x.test)

cat("Optimal lambda (Lasso):", cv.lasso$lambda.min, "\n")

# ROC for Lasso
roc(test_data$athetisi, as.numeric(lasso.probs),
    ci = TRUE, smooth = TRUE, plot = TRUE,
    xlab = '1-Specificity',
    main = 'ROC curve for lasso regression')

# --- Decision Trees ----------------------------------------------------------

# Classification tree
tree2 <- rpart(athetisi ~ ., data = training_data, method = 'class')
rpart.plot(tree2, extra = 106, main = 'Decision Tree')

# Pruned tree
pruned <- prune(tree2, cp = 0.01)
rpart.plot(pruned, extra = 106, main = 'Pruned Decision Tree')

# Tree predictions
tree.pred <- predict(pruned, newdata = test_data[, -5], type = "prob")[, 2]

# ROC for Classification Tree
roc(test_data$athetisi, tree.pred,
    ci = TRUE, smooth = TRUE, plot = TRUE,
    xlab = '1-Specificity',
    main = 'ROC curve for Classification Tree')

# --- Random Forest -----------------------------------------------------------

random.forest.model <- randomForest(athetisi ~ ., data = training_data,
                                     proximity = TRUE)

# Random Forest predictions
random.forest.pred <- predict(random.forest.model,
                               newdata = test_data[, -5],
                               type = "prob")[, 2]

# ROC for Random Forest
roc(test_data$athetisi, random.forest.pred,
    ci = TRUE, smooth = TRUE, plot = TRUE,
    xlab = '1-Specificity',
    main = 'ROC curve for Random Forest')

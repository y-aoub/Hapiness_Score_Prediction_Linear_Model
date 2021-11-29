install.packages("car") 
install.packages("lmtest")
install.packages("olsrr")
install.packages("caret")
install.packages("glmnet")

#Import the data frame (csv format) ------------------------------------------------------
df <- read.csv(file = '/home/cytech/Project_LM/Hapiness_Report_Data_2019.csv')
df <- df[,3:9] #removing the columns "Rank" and "Country" from the original data frame
head(df)

#Visualize df
plot(df)

#Model adjustment assessment --------------------------------------------------------------

#Fitting the model
initial_model <- lm(formula = Score ~ GDP_per_capita + Social_support + 
     Healthy_life_expectancy + Freedom_to_make_life_choices +
     Generosity + Perceptions_of_corruption, data = df) #fitting the linear regression 

initial_model #estimated coefficients and intercept
summary(initial_model) #adjusted R-squared:  0.7703

#Residual hypotheses validation ----------------------------------------------------------

library(car)
library(lmtest)
e <- residuals(initial_model)

#Hyp1: mean(residuals) = 0
mean(e) #computing the mean of the model's residuals --> Hyp1 validated (mean(e) = -4.896841e-17 ~ 0)

#Hyp2: variance(residuals) = cst
var(e) #computing the variance of the model's residuals
ncvTest(initial_model) #p (p-value) = 0.18261 > 5% --> Hyp2 validated (Breush-Pagan test proves homocscedasticity)

#Hyp3: autocorrelation of residuals
dwtest(initial_model) #p-value = 0.01097 < 5% --> Possibility of autocorrelation (Durbin-Watson test proves possibility of autocorrelation of residuals)
predicted_score <- predict(initial_model, df[,c("GDP_per_capita", "Social_support", 
                                                "Healthy_life_expectancy", "Freedom_to_make_life_choices",
                                                "Generosity", "Perceptions_of_corruption")])
plot(x=predicted_score, y=e, xlab = "Predicted_Score", 
     ylab = "Residuals") #the plot shows that there is no autocoerrelation of residuals --> Hyp3 validated

#Hyp4: normality of residuals
qqnorm(e, pch = 1, ylab="Standardized Residuals", frame = FALSE)
qqline(e, col = "steelblue", lwd = 2) #the plot shows that the residuals are aligned on the blue line --> Hyp4 validated

#Outliers detection ----------------------------------------------------------------------

#Student test

res.student <- rstudent(initial_model) #studentized residuals
res.student

alpha <- 0.1 #risk set to 10% (usually set to 10%)
threshold.student <- qt(1-alpha/2,156-6-2) #determination of the student threshold (ddl = n-p-2 = 156-6-2)

outlier.rstudent <- (res.student < -threshold.student | res.student > +threshold.student)
ab.student <- df[outlier.rstudent,] #selecting the outliers (the points with the condition outlier.rstudent)

plot(x = df$Score, y = res.student, cex=0.75) #plotting studentized residuals as a function of Score
abline(h = -threshold.student) #plotting -threshold.student
abline(h = +threshold.student) #plotting +threshold.student
abline(h = 0)
text(df$Score[outlier.rstudent],res.student[outlier.rstudent],
      rownames(df)[outlier.rstudent]) #indexing the outliers on the plot

#Leverage computing 

indicateurs <- influence.measures(initial_model)
attributes(indicateurs)
res.hat <- indicateurs$infmat[,"hat"] #leverage for each observation 

threshold.hat <- 2*(6+1)/156 #2*(p+1)/n

outlier.leverage <- (res.hat > threshold.hat) 
ab.hat <- df[atypiques.leverage,] #selecting the outliers (the points with the condition outlier.leverage)

plot(x = df$Score, y = res.hat, cex=0.75) #plotting leverages as a function of Score
abline(h = threshold.hat) #plotting +threshold.hat
text(df$Score[outlier.leverage],res.hat[outlier.leverage],
     rownames(df)[outlier.leverage]) #indexing the outliers on the plot

#Influential values detection ------------------------------------------------------------

#Cook's distance computing

library(olsrr)

ols_plot_cooksd_chart(initial_model) #plotting the Cook's distance for each observation
                                     #(computed Threshold = 0.026) to detect influential values

#Significance of each variable and significance of the whole model -----------------------

summary(initial_model)

#we see that all the variables are significant indicated by '***' and '**'
#with a small Pr(>|t|) (p-value), except "Generosity" and "Perception_of_corruption"
#which are not significant at all and can be removed from the model 

#we see that F-statistic (Fisher test) = 87.62 which is high and a small p-value
# (p-value: < 2.2e-16), hence the whole model is highly significant (at least one variable
#is significant)

#Multi-collinearity check -----------------------------------------------------------------

#Variance inflation factor (VIF) computing

vif_values <- vif(initial_model) #vif function from the package "car" computes VIF of each variable

barplot(vif_values, main = "VIF Values",
        horiz = TRUE, col = "steelblue") #create horizontal bar chart to display each VIF value 

vif_thresholder <- 4 #the threshold is set to 4 
abline(v = vif_thresholder, lwd = 3, lty = 2) #add vertical line at 4

#we see that the VIF(GDP_per_capita) exceeds the VIF threshold, then the variable is
#suspected to be behind a collinearity

cor(df[,2:7]) #correlation matrix of predictive variables
#the correlation matrix shows that the variable GDP_per_capita could be correlated 
#with Healthy_life_expectancy and Social_support

#Variables selection using AIC criteria (Backward selection)

library(olsrr)

ols_step_best_subset(initial_model)
#returns the 6 models with less AIC by doing a forward selection of variables
#hence the variable Generosity is the less significant which makes the model 5
#(without the variable generosity) the most performing with Adj. R-Square = 0.7703
#and the smallest AIC = 254.5400 among the other models

new_model <- lm(formula = Score ~ GDP_per_capita + Social_support +
                Healthy_life_expectancy + Freedom_to_make_life_choices +
                Perceptions_of_corruption, data = df) #initial_model - "Generosity"

summary(new_model) #we see that the whole model is more significant (high F-statistic = 105) than before
                   #as well as the variables of the model (small p-values)

#Cross-validation (repeated 10 times) -----------------------------------------------------

library(caret)

set.seed(123) 

train.control.k.fold <- trainControl(method = "repeatedcv", number = 10, repeats = 10) # Define training control: 10-Fold repeated 10 times
train.control.loocv <- trainControl(method = "LOOCV")

#LOOCV
initial_model_loocv <- train(Score ~ GDP_per_capita + Social_support +
                             Healthy_life_expectancy + Freedom_to_make_life_choices + Generosity +
                             Perceptions_of_corruption, data = df, method = "lm",
                             trControl = train.control.loocv) #training the initial_model (with all the predictive variables)

new_model_loocv <- train(Score ~ GDP_per_capita + Social_support +
                         Healthy_life_expectancy + Freedom_to_make_life_choices +
                         Perceptions_of_corruption, data = df, method = "lm",
                         trControl = train.control.loocv) #training the new_model (with all the variables except the variable "Generosity")

initial_model_loocv #RMSE_initial_model (prediction error)
new_model_loocv #RMSE_new_model (prediction error)

initial_model_10_fold <- train(Score ~ GDP_per_capita + Social_support +
                               Healthy_life_expectancy + Freedom_to_make_life_choices + Generosity +
                               Perceptions_of_corruption, data = df, method = "lm",
                               trControl = train.control.k.fold) #training the initial_model (with all the predictive variables)

new_model_10_fold <- train(Score ~ GDP_per_capita + Social_support +
                           Healthy_life_expectancy + Freedom_to_make_life_choices +
                           Perceptions_of_corruption, data = df, method = "lm",
                           trControl = train.control.k.fold) #training the new_model (with all the variables except the variable "Generosity")

initial_model_10_fold #RMSE_initial_model (prediction error) 
new_model_10_fold #RMSE_new_model (prediction error) 

#then we will choose the new_model over the initial_model, because in both cross validations
#LOOCV and 10_Fold RMSE_new_model < RMSE_initial_model

# Ridge & LASSO Regularization ------------------------------------------------------------

library(glmnet)

y <- df$Score
x <- data.matrix(df[, c('GDP_per_capita', 'Social_support', 'Healthy_life_expectancy',
                        'Freedom_to_make_life_choices', 'Perceptions_of_corruption')]) #define matrix of predictive variables

#Ridge Regularization

cv_model_Ridge <- cv.glmnet(x, y, alpha = 1) #perform K-Fold cross-validation to find optimal lambda value

best_lambda_Ridge <- cv_model_Ridge$lambda.min #find optimal lambda value that minimizes mean squared error (MSE)

plot(cv_model_Ridge) #produce plot of MSE by lambda value

new_model_Ridge <- glmnet(x, y, alpha = 1, lambda = best_lambda_Ridge) #find coefficients of best model (alpha = 1 --> Ridge reg)
coef(new_model_Ridge) #show the coefficients of the Ridge regression 

y_predicted_Ridge <- predict(new_model_Ridge, s = best_lambda_Ridge, newx = x) #use fitted best model to make predictions on the matrix x defined above
                                                                         
SST_1 <- sum((y - mean(y))^2) #compute SST_1
SSE_1 <- sum((y_predicted_Ridge - y)^2) #compute SSE_1

rsq_1 <- 1 - SSE_1/SST_1 #computing R-squared
rsq_1 #rsq_1 = 0.777623 is slightly better than the r-squared for the new_model (rsq = 0.7703) 

#LASSO Regularization 

cv_model_LASSO <- cv.glmnet(x, y, alpha = 0) #perform K-Fold cross-validation to find optimal lambda value

best_lambda_LASSO <- cv_model_LASSO$lambda.min #find optimal lambda value that minimizes mean squared error (MSE)

plot(cv_model_LASSO) #produce plot of MSE by lambda value

new_model_LASSO <- glmnet(x, y, alpha = 0, lambda = best_lambda_LASSO) #find coefficients of best model (alpha = 0 --> LASSO reg)
coef(new_model_LASSO) #shows the coefficients of the LASSO regression 

y_predicted_LASSO <- predict(new_model_LASSO, s = best_lambda_LASSO, newx = x) #use fitted best model to make predictions on the matrix x defined above
                                                                         
SST_2 <- sum((y - mean(y))^2) #compute SST_2
SSE_2 <- sum((y_predicted_LASSO - y)^2) #compute SSE_2

rsq_2 <- 1 - SSE_2/SST_2 #computing R-squared
rsq_2 #rsq_2 = 0.7753831 is slightly better than the r-squared for the new_model (rsq = 0.7703)


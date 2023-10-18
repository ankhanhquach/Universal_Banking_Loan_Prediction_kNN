# Lab kNN Khanh Quach

# kNN Lab

library(ggplot2)
library(caret)

# 1.0 Load Data
bank <- read.csv("UniversalBank.csv")
names(bank)
head(bank,10)
str(bank)

# Read file 
str(bank)

# 1.1 Clean up
# Drop ID and zip code columns.( Except ID and ZIP code- mentioned in the doc)

bank <- bank[ , -c(1, 5)]
names(bank)

# Reorder variables. Put the response last.

bank <- bank[ , c(1:7, 9:12, 8)]

head(bank,10)

# Set categorical variables as factor.

bank$Education <- as.factor(bank$Education)
bank$Securities.Account <- as.factor(bank$Securities.Account)
bank$CD.Account <- as.factor(bank$CD.Account) 
bank$Online <- as.factor(bank$Online) 
bank$CreditCard <- as.factor(bank$CreditCard) 

# Rename outcome variable values (optional).
# Note: We can do the problem in "0" and "1" or name them.

bank$Personal.Loan <- factor(bank$Personal.Loan,
                             levels = c("0", "1"),
                             labels = c("No", "Yes"))

table(bank$Personal.Loan)

# 1.2. Set training and validation sets
set.seed(666)

train_index <- sample(1:nrow(bank), 0.6 * nrow(bank))
valid_index <- setdiff(1:nrow(bank), train_index)

train <- bank[train_index, ]
valid <- bank[valid_index, ]

nrow(train)
nrow(valid)

# 4. Define new customer
new_cust <- data.frame(Age = 40,
                       Experience = 10,
                       Income = 84,
                       Family = 2,
                       CCAvg = 2,
                       Education = 2,
                       Mortgage = 0,
                       Securities.Account = 0,
                       CD.Account = 0,
                       Online = 1,
                       CreditCard = 1)

# Set categorical variables as factor.

new_cust$Education <- as.factor(new_cust$Education)
new_cust$Securities.Account <- as.factor(new_cust$Securities.Account)
new_cust$CD.Account <- as.factor(new_cust$CD.Account) 
new_cust$Online <- as.factor(new_cust$Online) 
new_cust$CreditCard <- as.factor(new_cust$CreditCard) 

new_cust

# 5.0 prepare for kNN. 

# Normalisation, only for numerical variables

train_norm <- train
valid_norm <- valid

norm_values <- preProcess(train[, -c(6, 8:12)],
                          method = c("center",
                                     "scale"))

# Then normalise the training and validation sets.
train_norm[, -c(6, 8:12)] <- predict(norm_values,
                                     train[, -c(6, 8:12)])
valid_norm[, -c(6, 8:12)] <-predict(norm_values,
                                    valid[, -c(6, 8:12)])
newcust_norm <- predict(norm_values, new_cust)
newcust_norm

# 7.0 Train kNN for predictions

# 7.1 k = 3
knn_model_k3 <- caret::knn3(Personal.Loan ~ ., 
                            data = train_norm, k = 3)
knn_model_k3

# Predict training set with k = 3 
knn_pred_k3_train <- predict(knn_model_k3, 
                             newdata = train_norm[, -c(12)], 
                             type = "class")
head(knn_pred_k3_train)

# Evaluate the confusion matrix with k = 3 
confusionMatrix(knn_pred_k3_train, as.factor(train_norm[, 12]),
                positive = "Yes")

# 7.2 k = 5 

# train k = 5
knn_model_k5 <- caret::knn3(Personal.Loan ~ ., 
                            data = train_norm, k = 5)
knn_model_k5

# Predict training set with k = 5
knn_pred_k5_train <- predict(knn_model_k5, 
                             newdata = train_norm[, -c(12)], 
                             type = "class")
head(knn_pred_k5_train)

# Evaluate the confusion matrix with k = 5 
confusionMatrix(knn_pred_k5_train, as.factor(train_norm[, 12]),
                positive = "Yes")

# 7.3 k = 7

# train k = 7
knn_model_k7 <- caret::knn3(Personal.Loan ~ ., 
                            data = train_norm, k = 7)
knn_model_k7

# Predict training set with k = 7
knn_pred_k7_train <- predict(knn_model_k7, 
                             newdata = train_norm[, -c(12)], 
                             type = "class")
head(knn_pred_k7_train)

# Evaluate confusion matrix with k = 7
confusionMatrix(knn_pred_k7_train, as.factor(train_norm[, 12]),
                positive = "Yes")

# 7.4 predict validation set

# use k = 3

# Predict training set with k = 3 
knn_pred_k3_valid <- predict(knn_model_k3, 
                             newdata = valid_norm[, -c(12)], 
                             type = "class")
head(knn_pred_k3_valid)

# Evaluate confusion matrix with k = 3 
confusionMatrix(knn_pred_k3_valid, as.factor(valid_norm[, 12]),
                positive = "Yes")

library(ROSE)
# Graph the ROC and show the AUC - number show @ the console for k = 3 
ROSE::roc.curve(valid_norm$Personal.Loan, 
                knn_pred_k3_valid)
# Area under the curve (AUC): 0.787

# use k = 5

# Predict training set with k = 5 
knn_pred_k5_valid <- predict(knn_model_k5, 
                             newdata = valid_norm[, -c(12)], 
                             type = "class")
head(knn_pred_k5_valid)

# Evaluate confusion matrix with k = 5
confusionMatrix(knn_pred_k5_valid, as.factor(valid_norm[, 12]),
                positive = "Yes")

# Graph the ROC and show the AUC - number show @ the console for k = 5
ROSE::roc.curve(valid_norm$Personal.Loan, 
                knn_pred_k5_valid)
# Area under the curve (AUC): 0.746

# use k = 7

# Predict training set for k = 7 
knn_pred_k7_valid <- predict(knn_model_k7, 
                             newdata = valid_norm[, -c(12)], 
                             type = "class")
head(knn_pred_k7_valid)

# Evaluate confusion matrix with k = 7
confusionMatrix(knn_pred_k7_valid, as.factor(valid_norm[, 12]),
                positive = "Yes")

# Graph the ROC and show the AUC - number show @ the console for k = 7
ROSE::roc.curve(valid_norm$Personal.Loan, 
                knn_pred_k7_valid)

# Area under the curve (AUC): 0.717

# Note on the sensitivity, specificity, precision, and AUC ROC

#knn_pred_k3_train 
#Sensitivity : 0.77273         
#Specificity : 0.99851 

#knn_pred_k5_train
#Sensitivity : 0.69156         
#Specificity : 0.99814

#knn_pred_k7_train
#Sensitivity : 0.60065         
#Specificity : 0.99963         

#knn_pred_k3_valid
#Sensitivity : 0.5756          
#Specificity : 0.9978

#knn_pred_k5_valid
#Sensitivity : 0.4942          
#Specificity : 0.9973 

#knn_pred_k7_valid
#Sensitivity : 0.4360          
#Specificity : 0.9989

# Sensitivity (tpr) decrease when k increase - 
# higher sensitivity is better -> k=3 is the best option.

# k = 3 train: 0.9753 and validation = 0.9615, AUC: 0.787
# k = 5 train: 0.9667 and validation = 0.954, AUC: 0.746
# k = 7 train: 0.9587 and validation = 0.9505, AUC: 0.717

# ROC AUC score shows how well the classifier distinguishes positive and negative classes. It can take values from 0 to 1. A higher ROC AUC indicates better performance. Therefore based on all of the above we can conclude that k=3 will be the highest accuracy model.

# k=3 provides the highest accuracy on the validation set, but it's also the most complex model lowest 'k'.

# 8. use kNN for new customer, k = ???

# k=3 has the best of both

# Use k = 3 for kNN new customer, k = 3

# Using k=3  to predict the new customer
knn_pred_new_cust <- predict(knn_model_k3,newdata = newcust_norm, type = "class")
knn_pred_new_cust

# 2688 cases were correctly predicted as "No" and  
# 238 cases were correctly predicted as “Yes”. 

# The accuracies of the predictions on the training and validation 
# sets are  both high (0.9753 VS 0.9615) for k = 3, which do not suggest overfitting

# 9. Answers

# The result for the new customer in regards to Personal.Loan is “No”.
# We can assume that the new customer is not likely to accept the loan offer.




################################################################
##### Big Data HW 3 ####################################
##### By Blain Morin ##########################
############################################

require(RSQLite) == T || install.packages("RSQLite")
require(tibble) == T || install.packages("tibble")
require(dplyr) == T || install.packages("dplyr")
require(caret) == T || install.packages("caret")
require(doParallel) || install.packages("doParallel")
require(readr) == T || install.packages("readr")

authors <- function() {
  c("Blain Morin")
}

################################################################
#### Question 1 #######################################
################################################

library(doParallel)
library(readr)
library(RSQLite)
library(tibble)
library(caret)
library(RSQLite)
library(tibble)
library(dplyr)

db = dbConnect(SQLite(), dbname="pred.sqlite")

## Get pred IDs
pred.id = as.tibble(
  
  dbGetQuery(conn = db, "SELECT id FROM pred")
  
)

## Get demo IDs
demo.id = as.tibble(
  
  dbGetQuery(db, "SELECT id FROM demo")
  
)

## Get Outcome IDs
outcome.id = as.tibble(
  
  dbGetQuery(db, "SELECT id FROM outcome")
  
)

## Get IDs that are in pred AND demo
pred.demo.id = semi_join(pred.id, demo.id)


## Get IDs that are in all three tables
all.id = semi_join(pred.demo.id, outcome.id)


## Write csv for IDs in all tables
write.csv(all.id, file = "idall.csv", row.names = FALSE)


## Get IDs in pred and demo, but missing from outcome
missing.id = anti_join(pred.demo.id, outcome.id)


## Write csv for missing IDs
write.csv(missing.id, file = "idmissing.csv", row.names = FALSE)



############################################################
#### Question 2 ##############################
####################################


## Change database tables to tibbles
outcome.table = as.tibble(dbGetQuery(db, "SELECT * FROM outcome"))
pred.table = as.tibble(dbGetQuery(db, "SELECT * FROM pred"))
demo.table = as.tibble(dbGetQuery(db, "SELECT * FROM demo"))

## Prepare our training data
data.q2 = inner_join(all.id, pred.table)
data.q2 = inner_join(data.q2, demo.table, by = "id")
data.q2 = inner_join(data.q2, outcome.table, by = "id")
data.q2$gender = ifelse(data.q2$gender == "F", 1, 0)
data.q2$age = as.numeric(data.q2$age)
data.q2 = as.matrix(data.q2)

##Prepare our data for prediction
p.q2 = inner_join(missing.id, pred.table)
p.q2 = inner_join(p.q2, demo.table, by = "id")
p.q2$gender = ifelse(p.q2$gender == "F", 1, 0)
p.q2$age = as.numeric(p.q2$age)
p.q2 = as.matrix(p.q2)


#######################
### Model Tuning ###
##################

set.seed(1)

## Create list of alphas and lambdas to check
## Here alpha sequences between ridge, elastic, and lasso
parameter.values = expand.grid(alpha = seq(0,1, by = .5) , 
                               lambda = 10^seq(-3, 3, length.out = 300))


## Train the model
model.q2 = train(data.q2[ , 2:108], y = data.q2[ , 109], 
                 method = "glmnet", tuneGrid = parameter.values)


## Predict the missing outcomes
predictions.q2 = predict.train(model.q2, newdata = p.q2[ , 2:108])


## Export csv
output1 = data.frame(id = p.q2[,1], predicted = predictions.q2)
write.csv(output1, file = "output1.csv")




###############################################################
#### Question 3 ################################
########################################

## Set up parallelization 
cl = makeCluster(4)
registerDoParallel(cl)
set.seed(1)


## Create outcome matrix
outcomes = data.q2[ , 109:126]


## Create list of parameters to check
parameter.values.q3 = expand.grid(alpha = seq(0,1, by = .5), lambda = 10^seq(3, -3, length.out = 300))


## Train on each outcome and predict
output2 = foreach(i = 1:ncol(outcomes), .combine = cbind, .packages = "caret") %dopar% {
  
  lasso = train(data.q2[ , 2:108], y = outcomes[ , i],
                method = "glmnet", tuneGrid = parameter.values.q3)
  lasso.predictions = predict.train(lasso, newdata = p.q2[ , 2:108])
  
  
}


## Prepare for export
output2 = data.frame(id = missing.id, output2)
names(output2)[2:19] = names(outcome.table)[2:19]

write.csv(output2, file = "output2.csv")



################################################################
#### Question 4 ##############################
##################################

pred2 = read_csv("pred2.csv", col_names = FALSE)

## Add new predictors to data
names(pred2)[1] = "id"

data.q4 = inner_join(all.id, pred.table)
data.q4 = inner_join(data.q4, demo.table, by = "id")
data.q4 = inner_join(data.q4, pred2, by = "id")
data.q4 = inner_join(data.q4, outcome.table, by = "id")
data.q4$gender = ifelse(data.q4$gender == "F", 1, 0)
data.q4$age = as.numeric(data.q4$age)
data.q4 = as.matrix(data.q4)



##Prepare our data for prediction
p.q4 = inner_join(missing.id, pred.table)
p.q4 = inner_join(p.q4, demo.table, by = "id")
p.q4 = inner_join(p.q4, pred2, by = "id")
p.q4$gender = ifelse(p.q4$gender == "F", 1, 0)
p.q4$age = as.numeric(p.q4$age)
p.q4 = as.matrix(p.q4)


## Create outcome matrix
outcomes = data.q2[ , 109:126]


## Create list of parameters to check
parameter.values.q4 = expand.grid(alpha = seq(0,1, by = .5), lambda = 10^seq(3, -3, length.out = 300))


## Train on each outcome and predict
output3 = foreach(i = 1:ncol(outcomes), .combine = cbind, .packages = "caret") %dopar% {
  
  lasso = train(data.q4[ , 2:307], y = outcomes[ , i],
                method = "glmnet", tuneGrid = parameter.values.q4)
  lasso.predictions = predict.train(lasso, newdata = p.q4[ , 2:307])
  
  
}

### Prepare for export
output3 = data.frame(id = missing.id, output3)
names(output3)[2:19] = names(outcome.table)[2:19]

write.csv(output3, file = "output3.csv")

## Yes, the added predictors decreased RMSE.













####################################################
##                                                ##
##        Ames Housing Analysis                   ##
##                                                ##
####################################################

library(tidyverse)
library(rsample)
library(InformationValue)
library(GGally)
library(caret)

setwd('~/Data Analysis R & Python/Homework/HW 6')

ames <- read_csv('ames_housing.csv')

view(ames)

#####################################################
######### (1) Feature engineering

# is there a premium for higher condition homes built more recently?
ames <- ames %>%
  mutate(cond_year_built = `Overall Cond` * `Year Built`)


#####################################################
########## (2) Univariate (one dep var) linear regression 

mod_uni <- lm(SalePrice ~ `Year Built`, data = ames)


#####################################################
########## (3) Multivariate (many dep vars) linear regression 

mod_multi <- lm(SalePrice ~ `Overall Cond` + `Year Built` + cond_year_built,
               data = ames)


#####################################################
########## (4) Summary stats about linear regression

summary(mod_uni)
summary(mod_multi)

summary(mod_uni)$r.squared
# R2 is 0.31, which means only about 31% of variation in sale price can be
# explained by the year that the house was built in.

summary(mod_multi)$r.squared
# R2 is 0.35, which means only about 35% of variation in sale price can be
# explained by house condition, year the house was built in, and the product of 
# those 2 features.

modelr::rmse(mod_uni, ames)
# rmse = $66259.04
modelr::rmse(mod_multi, ames)
# rmse = $64164.08


#####################################################
########## (5) Logistic regression

ames_subset <- ames %>% select(SalePrice, `Overall Cond`, `Year Built`)

set.seed(42)

split <- initial_split(ames_subset, prop = 0.7)
train <- training(split) # extract the actual data 
test <- testing(split) # extract the actual data

dim(split)
dim(train)
dim(test)

train_dummy <- train %>% mutate(expensive = ifelse(SalePrice > 200000, 1, 0))

mod_log <- glm(expensive ~ `Overall Cond` + `Year Built`, family="binomial", data=train_dummy)

table(train_dummy$expensive)

#####################################################
########## (6) Metrics specific to classification models

summary(mod_log)

varImp(mod_log)
# Year built appears to be more important than overall condition in this model.

predicted <- predict(mod_log, test, type="response")

test_dummy <- test %>% mutate(expensive = ifelse(SalePrice > 200000, 1, 0))
optimalCutoff(test_dummy$expensive, predicted)
# optimal cutoff is 0.44 rather than 0.5

# ROC curve
plotROC(test_dummy$expensive, predicted)
# although this might not be great metric since this dataset is imbalanced


#####################################################
########## (7) Charts and graphs

ggpairs(ames_subset)
# shows that there's a moderate positive correlation between year built and sale
# price such that modern houses sell at higher prices than older homes

options(scipen=999)

ggplot(test_dummy, aes(`Year Built`, SalePrice)) + 
  geom_point(aes(colour = `Overall Cond`))

# there doesn't seem to be a strong interaction between year built and overall
# condition on sale price. there doesn't seem to be a huge premium for higher
# condition homes built more recently. however, our dataset also doesn't seem to
# contain many modern homes in top condition. this would be something to further
# explore!


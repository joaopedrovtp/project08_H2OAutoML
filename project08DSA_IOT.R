### Project 08 - IoT - Electrical Energy Prediction with H2OAutoMl

setwd("C:/FCD/04.MachineLearning/Projetos/Projeto08")
getwd()

library(caret)
library(ggplot2)
library(knitr)
library(plyr)
library(dplyr)
library(corrplot)
library(plotly)
library(randomForest)
library(h2o)

# Loading datasets
df_train <- read.csv("projeto8-training.csv")
df_test <- read.csv("projeto8-testing.csv")

# Merging test and train dataframes
df_train$label <- "train" # creating column for label
df_test$label <- "test"
df_merged <- rbind(df_train, df_test)

# Showing some infos about the data 
View(df_merged)
str(df_merged)
summary(df_merged)

# Converting varible 'date' forma to date&hour type
df_merged$date <- as.POSIXct(df_merged$date) 

# Checking missing values
sapply(df_train, function(x) sum(is.na(x)))

# Create dataframe with numeric features only
num_cols_df <- df_merged[,-c(1,31,32,33)]


####### Exploratory Analysis #######

## Target variable 'Appliances' 
# Histogram
hist(df_merged$Appliances, breaks = 30,  col='darkblue',
     xlab="Energy consumption", main = "Frequencies - Energy consumption")

# Unique values from Appliances
sort(unique(df_merged$Appliances))

# Top Frequencies table
kable(table(cut(df_merged$Appliances, breaks=seq(1, 201, 10))), align = 'c')

# Plot time series Energy consumption - Each point measured
plot_ly(x = df_merged$date , y = df_merged$Appliances, type="scatter", mode="markers")


# We can see the target variable is left-skewed. Let's keep eye on that for later transformation.

# Correlation plot
m_cor <- cor(num_cols_df)
corrplot(m_cor, method = "square",tl.col = "black")


# Here we see the temperatures are positive correlated 
# from each other, and humidity too. But we can see a different behaviour   
# with the RH_6 by temperatures with  negative correlation. 
# Maybe it's a part of the house that has major impact from temperatures from others. 
# Also, note that rv1 and rv2 are highly correlated, it seems the same. Let's check it later
# and remove one of them if applicable.


# Temperatures frequencies

par(mfrow = c(3,3))
hist(df_merged$T1, breaks = 30,  col='darkred', xlab = "", main = "T1")
hist(df_merged$T2, breaks = 30,  col='darkred', xlab = "", main = "T2")
hist(df_merged$T3, breaks = 30,  col='darkred', xlab = "", main = "T3")
hist(df_merged$T4, breaks = 30,  col='darkred', xlab = "", main = "T4")
hist(df_merged$T5, breaks = 30,  col='darkred', xlab = "", main = "T5")
hist(df_merged$T6, breaks = 30,  col='darkred', xlab = "", main = "T6")
hist(df_merged$T7, breaks = 30,  col='darkred', xlab = "", main = "T7")
hist(df_merged$T8, breaks = 30,  col='darkred', xlab = "", main = "T8")
hist(df_merged$T9, breaks = 30,  col='darkred', xlab = "", main = "T9")


# RH - Relative humidities frequencies
hist(df_merged$RH_1, breaks = 30,  col='#0c5069', xlab = "", main = "RH_1")
hist(df_merged$RH_2, breaks = 30,  col='#0c5069', xlab = "", main = "RH_2")
hist(df_merged$RH_3, breaks = 30,  col='#0c5069', xlab = "", main = "RH_3")
hist(df_merged$RH_4, breaks = 30,  col='#0c5069', xlab = "", main = "RH_4")
hist(df_merged$RH_5, breaks = 30,  col='#0c5069', xlab = "", main = "RH_5")
hist(df_merged$RH_6, breaks = 30,  col='#0c5069', xlab = "", main = "RH_6")
hist(df_merged$RH_7, breaks = 30,  col='#0c5069', xlab = "", main = "RH_7")
hist(df_merged$RH_8, breaks = 30,  col='#0c5069', xlab = "", main = "RH_8")
hist(df_merged$RH_9, breaks = 30,  col='#0c5069', xlab = "", main = "RH_9")
par(mfrow = c(1,1)) # Back to initial grid config  

## Histograms of T_out by WeekStatus

# Data division for weekend and weekday  
weekend <- df_merged %>% 
  filter(WeekStatus == "Weekend")

weekday <- df_merged %>% 
  filter(WeekStatus == "Weekday")

# Hist plot
hist(weekend$T_out, ylim = c(0,1300), breaks=30, col=rgb(1,0,0,0.5), xlab="Temperature outside", 
     ylab="Frequency", main="Distribution of Out temperatures by Weekend and Weekday" )

# Second with add=T to plot on top
hist(weekday$T_out, ylim = c(0,1300) ,breaks=30, col=rgb(0,0,1,0.5), add=T)

# Add legend
legend("topright", legend=c("weekend","weekday"), col=c(rgb(1,0,0,0.5), 
                                                      rgb(0,0,1,0.5)), pt.cex=2, pch=15)


## Examining the tdewpoint
# Group date by day and summarize mean of tdewpoint  
dewp_time <- df_merged %>% 
  group_by(date = as.Date(date)) %>%
  summarize(tot = mean(Tdewpoint))

plot_ly(x = dewp_time$date , y = dewp_time$tot, type="bar", color= "red")

# Lights - we see most of registers are 0w
ggplot(data=df_merged, aes(x=lights)) +
  geom_bar(fill= '#E69F00') + 
  ylab("Count") + xlab("Lights (Wh) ") + 
  ggtitle("Lights by Wh")


####### Feature engineering #######

# Checking and removing identical features
features_pair <- combn(names(num_cols_df), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(num_cols_df[[f1]] == num_cols_df[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

# Removing rv2 
df_merged <- subset(df_merged, select = -c(rv2))


# Log-transform to target variable 
log_Appliances <- log(df_merged$Appliances)

hist(log_Appliances, breaks = 30,  col='darkblue',
     xlab="Energy consumption", main = "Frequencies - Energy consumption")

# A huge difference from the first left-skewed histogram

# Normalize data
num_cols_df <- subset(num_cols_df, select = -c(Appliances, rv2))
scaled_num_cols <- scale(num_cols_df)

# One hot enconding - WeekStatus feature
categorical_f <- subset(df_merged, select = c(WeekStatus))
dmy <- dummyVars(" ~ .", data = categorical_f)
categorical_enc <- data.frame(predict(dmy, newdata = categorical_f))

# Feature Selection
importance <- randomForest(log_Appliances ~ ., 
                            data = scaled_num_cols, 
                            ntree = 100, 
                            nodesize = 10, importance = TRUE)

varImpPlot(importance)

# Interesting! NSM is considered the most valuable feature.
# The measures are MSE (Mean Squared error) and mean decrease in node impurity. 
# Let's now choose the best features analyzed.

 
# Removing variables rv1, T6, RH_6, T_out, RH_out and Visibility because
# they showed less importance related to Energy consumption.

scaled_num_cols <- subset(scaled_num_cols, select = -c(rv1, T6, RH_6, T_out, RH_out, Visibility))
df_final <- cbind(log_Appliances, scaled_num_cols, categorical_enc, df_merged['label'])
View(df_final)


####### ML Modeling  ####### 


# Train and test Split
# Get train data and remove last column 'label'
train_d <- df_final[df_final$label == 'train', !names(df_final) %in% c("label")] 

## Get test data and remove last column 'label'
test_d <- df_final[df_final$label == 'test', !names(df_final) %in% c("label")] 


# I'll use H20AutoML here. It can be used for automating the machine learning workflow
# and use mostly Stacked Ensembles models for training.
# For more information I recommend acess the documentation on the following link
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

h2o.init()

# Transforming data to a H2OFrame object
train_frame = as.h2o(train_d)
test_frame = as.h2o(test_d)
y <- "log_Appliances"

# When using H2OAutoml, the parameter 'training_frame' consider 
# all columns except the response as predictors, so we can skip setting
# the `x` argument explicitly. Here I'm using parameters:
# 'max_models' to determine the max number, excluding the Stacked Ensemble models.
# for test you can specify max time by 'max_runtime_secs' parameter (default unlimited time)
#

aml <- h2o.automl(y = y,
                  training_frame = train_frame,
                  max_models = 10,
                  max_runtime_secs = 1200,
                  seed = 36,
                  keep_cross_validation_predictions = TRUE,
                  project_name = "energy_cons_lb_frame")



# View the AutoML Leaderboard
lb <- aml@leaderboard
print(lb, n = nrow(lb))  

# Perfomance scores on test data
perf <- h2o.performance(aml@leader, test_frame)
perf

# Predictions (H2oautoML uses the leader model)
pred <- h2o.predict(aml, test_frame)
pred

# Convert predictions to original scale
predictions <- exp(pred)

# to Dataframe predictions
predictions <- as.vector(predictions)
df_result <- data.frame(predictions)
head(df_result)

# Saving model
h2o.saveModel(object = aml@leader, path=getwd())

# Saving prediction
write.csv(df_result, "predictions.csv")









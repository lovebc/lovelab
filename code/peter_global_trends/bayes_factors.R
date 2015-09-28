library(BayesFactor)

setwd("/Users/petersebastianriefer/Dropbox/PhD/Explore_Exploit/Up_Down_Trends/Analysis")

data<-read.csv('data_bayes.csv', header=TRUE)
attach(data)

# Exploration
explore_std<-aov(explore~condition)
summary(explore_std)
explore_bayes<-anovaBF(explore~condition, data)
summary(explore_bayes)

# Accuracy
accuracy_std<-aov(score~condition)
summary(accuracy_std)
accuracy_bayes<-anovaBF(score~condition, data)
summary(accuracy_bayes)

# Slopes
slope_std<-aov(slope~condition)
summary(slope_std)
slope_bayes<-anovaBF(slope~condition, data)
summary(slope_bayes)
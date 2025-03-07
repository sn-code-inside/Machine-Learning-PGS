#############################################
### Kapitel 6.5 Neuronale Netzwerkmodelle ###
#############################################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('mlr3')
#install.packages('mlr3learners')
#install.packages('mlr3viz')
#install.packages('mlr3tuning')
#install.packages('neuralnet')

## Pakete laden
library(mlr3)
library(mlr3learners)
library(tidyverse)
library(neuralnet)

## Seed setzen
set.seed(42)

## Daten laden
data('BostonHousing', package = 'mlbench')

BostonHousing$chas = as.numeric(BostonHousing$chas)
# die Variable chas wird in eine numerische umgewandelt

which(apply(BostonHousing,2,function(x) any(is.na(x))))
# zeigt an in welchen Spalten des Datensatzes NAs vorkommen


## Daten in Trainings- & Testdaten aufteilen
train.set.bh = sample(nrow(BostonHousing), nrow(BostonHousing)/3*2)
# Zieht zufällig 2/3 der Reihen aus BostonHousing
test.set.bh = setdiff(1:nrow(BostonHousing), train.set.bh)
# Sucht von 1 bis Anzahl an Reihen in BostonHousing, welche nicht im Trainingsset vorkommen
train.data = BostonHousing[train.set.bh,]
# Speichert Trainingsdaten in seperatem Datensatz
test.data = BostonHousing[test.set.bh,]
# Speichert Testdaten in seperatem Datensatz


## Task festlegen
task = TaskRegr$new(id='boston', backend=BostonHousing, target='medv')

## Neuronales Netzwerkmodell
# Learner definieren
learner <- lrn("regr.nnet", size = 5, maxit = 100)

# Modell trainieren
learner$train(task, row_ids = train.set.bh)

# Vorhersagen
predictions <- learner$predict(task, row_ids = test.set.bh)

## Evaluation
predictions$score(msr('regr.mse'))
predictions$score(msr('regr.sae'))

# Visualisierung
autoplot(predictions)

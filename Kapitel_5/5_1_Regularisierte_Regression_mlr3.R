#############################################
### Kapitel 6.1 Regularisierte Regression ###
#############################################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('mlr3')
#install.packages('mlr3learners')
#install.packages('mlr3viz')
#install.packages('mlr3tuning')
#install.packages('carData')
#install.packages('glmnet')

## Pakete laden
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(tidyverse)
library(glmnet)
library(mlr3tuning)

## Seed setzen
set.seed(42)

## Daten laden
data('Wong', package='carData')
??Wong
# Zeigt Dokumentation f�r den Datensatz

Wong$sex = as.numeric(Wong$sex)
# die Variable sex wird in eine numerische Variable umgewandelt, da im Regressions-Task keine Faktoren als Features akzeptiert werden


## Daten in Trainings- & Testset aufteilen
train.set.Wong = sample(nrow(Wong), nrow(Wong)/3*2)
# Zieht zufällig 2/3 der Reihen aus Wong
test.set.Wong = setdiff(1:nrow(Wong), train.set.Wong)
# Zieht aus der Anzahl der Reihen in Wong, diejenigen die nicht im Trainingsset vorkommen


## Task definieren
task.verbal = TaskRegr$new(id='verbal', backend=Wong, target='viq')
# Definiert einen neuen Regressionstask
# id bezeichnet den Namen des Tasks;
# backend legt die Daten fest für die der Task verwendet werden soll;
# target legt die Targetvariable fest


## LASSO Regression 
# Learner definieren
learner.verbal.L = lrn('regr.glmnet', predict_type='response', alpha=1)
# alpha legt fest ob Lasso (alpha = 1) oder Ridge (alpha = 0) Regression

# Modell trainieren
model.verbal.L = learner.verbal.L$train(task=task.verbal, row_ids=train.set.Wong)
# task = gibt an mit welchem Task das Modell trainiert wird
# Mit row_ids wird festgelegt mit welchem Teil der Daten trainiert werden soll

# Vorhersage 
predictions.verbal.L = learner.verbal.L$predict(task=task.verbal, row_ids=test.set.Wong)
# Speichert die Vorhersagen für das Testset des Modells

# Evaluation von Mean-Squared-Error und Sum of absolute Errors
predictions.verbal.L$score(msr("regr.mse"))
predictions.verbal.L$score(msr("regr.sae"))

# Visualisierung
autoplot(predictions.verbal.L)


## Ridge-Regression
# Learner definieren
learner.verbal.R = lrn('regr.glmnet', predict_type='response', alpha=0)
# alpha legt fest ob Lasso (alpha = 1) oder Ridge (alpha = 0) Regression

# Modell trainieren
model.verbal.R = learner.verbal.R$train(task=task.verbal, row_ids=train.set.Wong)
# task = gibt an mit welchem Task das Modell trainiert wird
# Mit row_ids wird festgelegt mit welchem Teil der Daten trainiert werden soll

# Predictions 
predictions.verbal.R = learner.verbal.R$predict(task=task.verbal, row_ids=test.set.Wong)
# Speichert die Vorhersagen für das Testset des Modells

# Evaluation von Mean-Squared-Error und Sum of absolute Errors
predictions.verbal.R$score(msr('regr.mse'))
predictions.verbal.R$score(msr('regr.sae'))

# Visualisierung
autoplot(predictions.verbal.R)


## Parametertuning

# Learner festlegen
learner.verbal = lrn('regr.glmnet', predict_type='response')

# Parameter festlegen
params.verbal = ParamSet$new(list(
  alpha = p_int(lower=0, upper=1),
  lambda = p_int(lower=0, upper=10)))
# lower & upper legen die Unter- bzw. Obergrenze fest
# lambda ist der Regularisierungsfaktor 

# Festlegen wann Tuning beendet wird
terminator.verbal = trm('evals', n_evals=10)
# n_evals legt fest wieviele Runden evaluiert werden soll

# Resampling und Evaluationsmaß festlegen
resample.verbal = rsmp('cv', folds=10)
# cv = crossvalidation
# folds legt die Anzahl der Falten in der Kreuzvalidierung fest
measure.verbal = msr('regr.mse')

# Tuning Instance definieren
instance.verbal = TuningInstanceSingleCrit$new(
  task=task.verbal,
  learner=learner.verbal,
  resampling=resample.verbal,
  measure=measure.verbal,
  search_space=params.verbal,
  terminator=terminator.verbal
)
# Es werden die vorher festgelegten Parameter und Maße etc. übergeben

#Tunen
tuner.verbal.grid = tnr('grid_search')
tuner.verbal.grid$optimize(instance.verbal)

# Ergebnisse des Tunings auslesen
instance.verbal$result_learner_param_vals

# Hyperparameter des Learners mit den Ergebnissen festlegen
learner.verbal.tuned = learner.verbal
learner.verbal.tuned$param_set$values = instance.verbal$result_learner_param_vals

# nochmal mit besseren Parametern trainieren
learner.verbal.tuned$train(task.verbal, row_ids=train.set.Wong)

# neue Vorhersage
prediction.verbal.tuned = learner.verbal.tuned$predict(task.verbal, row_ids=test.set.Wong)
prediction.verbal.tuned$score(msr('regr.mse'))

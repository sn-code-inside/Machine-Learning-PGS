###########################################
### Kapitel 6.4 Support Vector Machines ###
###########################################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('mlr3')
#install.packages('mlr3learners')
#install.packages('mlr3viz')
#install.packages('mlr3tuning')
#install.packages('dplyr')

## Packete laden
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(tidyverse)
library(mlr3tuning)
library(dplyr)

## Seed setzen
set.seed(42)

## Daten laden
data('BEPS', package='carData')
??BEPS
# Zeigt Dokumentation für den Datensatz

# Feature gender umcodieren
BEPS$gender <- case_when(BEPS$gender == 'female' ~ 1,
                         BEPS$gender == 'male' ~ 2,
                         BEPS$gender == 'other' ~ 3)


## Daten in Trainings- & Testset aufteilen
train.set.BEPS = sample(nrow(BEPS), nrow(BEPS)/3*2)
# Zieht zufällig 2/3 der Reihen aus BEPS
test.set.BEPS = setdiff(1:nrow(BEPS), train.set.BEPS)
# Zieht aus der Anzahl der Reihen in BEPS, diejenigen die nicht im Trainingsset vorkommen


## Task definieren
task.vote = TaskClassif$new(id='vote', backend=BEPS, target='vote')
# Definiert einen neuen Klassifikationstask
# id bezeichnet den Namen des Tasks;
# backend legt die Daten fest für die der Task verwendet werden soll;
# target legt die Targetvariable fest

## Support Vector Machine
# Learner definieren
learner.vote = lrn('classif.svm', kernel="radial", type='C-classification')

# Modell trainieren
model.vote = learner.vote$train(task=task.vote, row_ids=train.set.BEPS)
# task = gibt an mit welchem Task das Modell trainiert wird
# Mit row_ids wird festgelegt mit welchem Teil der Daten trainiert werden soll

# Predictions
predictions.vote = learner.vote$predict(task=task.vote, row_ids=test.set.BEPS)
# Speichert die Vorhersagen für das Testset des Modells

## Evaluation
predictions.vote$confusion
predictions.vote$score(msr('classif.acc'))
predictions.vote$score(msr('classif.ce'))

# Visualisierung
autoplot(predictions.vote)


## Parametertuning

# Parameter festlegen
params.vote = ParamSet$new(list(
  cost = p_int(lower=1, upper=10),
  gamma = p_int(lower=1, upper=10)))
# lower & upper legen die Unter- bzw. Obergrenze fest
# cost legt die größe der Marings fest
# gamma bestimmt den Einfluss einzelner Datenpunkte auf die Decision Boundary

# Festlegen wann Tuning beendet wird
terminator.vote = trm('evals', n_evals=20)
# n_evals legt fest wieviele Runden evaluiert werden soll

# Resampling und Evaluationsmaß festlegen
resample.vote = rsmp('cv', folds=10)
# cv = crossvalidation
# folds legt die Anzahl der Falten in der Kreuzvalidierung fest
measure.vote = msr('classif.acc')

# Tuning Instance definieren
instance.vote = TuningInstanceSingleCrit$new(
  task=task.vote,
  learner=learner.vote,
  resampling=resample.vote,
  measure=measure.vote,
  search_space=params.vote,
  terminator=terminator.vote
)
# Es werden die vorher festgelegten Parameter und Maße etc. übergeben

#Tunen
tuner.vote.random = tnr('random_search') 
# hier wird festgelegt mit welcher Methode der Parameterraum durchsucht werden soll
# random_reach für zufälliges Suchen
tuner.vote.random$optimize(instance.vote)

# Ergebnisse des Tunings auslesen
instance.vote$result_learner_param_vals

# Hyperparameter des Learners mit den Ergebnissen festlegen
learner.vote.tuned = learner.vote
learner.vote.tuned$param_set$values = instance.vote$result_learner_param_vals

# nochmal mit besseren Parametern trainieren
learner.vote.tuned$train(task.vote, row_ids=train.set.BEPS)

# neue Vorhersage
prediction.vote.tuned = learner.vote.tuned$predict(task.vote, row_ids=test.set.BEPS)
prediction.vote.tuned$score(msr('classif.acc'))

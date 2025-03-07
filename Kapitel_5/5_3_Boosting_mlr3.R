############################
### Kapitel 6.3 Boosting ###
############################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('mlr3')
#install.packages('mlr3learners')
#install.packages('mlr3viz')
#install.packages('mlr3tuning')
#install.packages('xgboost')

## Pakete laden
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(tidyverse)
library(xgboost)
library(mlr3tuning)

## Seed setzen
set.seed(42)

## Daten laden
data('Freedman', package='carData')
crime = Freedman %>% filter_all(all_vars(!is.na(.)))
# Filtert Datensatz, dass keine NAs mehr drin sind

# Blick in die Daten
View(crime)
str(crime) 
# zeigt Struktur des Datensatzes


## Daten in Trainings- & Testdaten aufteilen
train.set.crime = sample(nrow(crime), nrow(crime)/3*2)
# Zieht zufällig 2/3 der Reihen aus dem Datensatz crime
test.set.crime = setdiff(1:nrow(crime), train.set.crime)
# Zieht aus der Anzahl der Reihen in crime, diejenigen die nicht im Trainingsset vorkommen


## Task festlegen
task.crime = TaskRegr$new(id='crime', backend=crime, target='crime')
# Definiert einen neuen Regressionstask
# id bezeichnet den Namen des Tasks;
# backend legt die Daten fest für die der Task verwendet werden soll;
# target legt die Targetvariable fest

## Boosting
# Learner definieren
learner.crime = mlr_learners$get('regr.xgboost')
learner.crime$param_set$values =  list(eta=.2, nrounds=25, verbose=1)
# legt die Parameter des Learners fest
# eta legt die Shrinkage Schrittgöße fest
# nrounds legt die Maximale Anzahl an Boosting Runden fest
# verbose legt die Ausführlichkeit fest, mit der Meldungen gezeigt werden; 0 = silent, 1 = warning, 2 = info, and 3 = debug

# Modell trainieren
model.crime = learner.crime$train(task=task.crime, row_ids=train.set.crime)
# task = gibt an mit welchem Task das Modell trainiert wird
# Mit row_ids wird festgelegt mit welchem Teil der Daten trainiert werden soll

# Vorhersagen
predictions.crime = learner.crime$predict(task=task.crime, row_ids=test.set.crime)
# Speichert die Vorhersagen für das Testset des Modells

## Evaluation
predictions.crime$score(msr('regr.mse'))
predictions.crime$score(msr('regr.sae'))

# Visualisierung
autoplot(predictions.crime)


## Parametertuning

# Parameter festlegen
params.crime = ParamSet$new(list(
  eta = p_dbl(lower=.1, upper=.5),
  min_child_weight = p_int(lower=1, upper=10)))
# lower & upper legen die Unter- bzw. Obergrenze fest
# eta entspricht der Lernrate
# min_chld_weight legt die mindest Summe der Gewichte pro Kinderknoten fest

# Festlegen wann Tuning beendet wird
terminator.crime = trm('evals', n_evals=10)
# n_evals legt fest wieviele Runden evaluiert werden soll

# Resampling und Evaluationsmaß festlegen
resample.crime = rsmp('cv', folds=10)
# cv = crossvalidation
# folds legt die Anzahl der Falten in der Kreuzvalidierung fest
measure.crime = msr('regr.mse')

# Tuning Instance definieren
instance.crime = TuningInstanceSingleCrit$new(
  task=task.crime,
  learner=learner.crime,
  resampling=resample.crime,
  measure=measure.crime,
  search_space=params.crime,
  terminator=terminator.crime
)
# Es werden die vorher festgelegten Parameter und Maße etc. übergeben

#Tunen
tuner.crime.random = tnr('random_search') 
# hier wird festgelegt mit welcher Methode der Parameterraum durchsucht werden soll
# random_reach für zufälliges Suchen
tuner.crime.random$optimize(instance.crime)

# Ergebnisse des Tunings auslesen
instance.crime$result_learner_param_vals

# Hyperparameter des Learners mit den Ergebnissen festlegen
learner.crime.tuned = learner.crime
learner.crime.tuned$param_set$values = instance.crime$result_learner_param_vals

# nochmal mit besseren Parametern trainieren
learner.crime.tuned$train(task.crime, row_ids=train.set.crime)

# neue Vorhersage
prediction.crime.tuned = learner.crime.tuned$predict(task.crime, row_ids=test.set.crime)
prediction.crime.tuned$score(msr('regr.mse'))

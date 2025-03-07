#################################
### Kapitel 6.2 Random Forest ###
#################################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('mlr3')
#install.packages('mlr3learners')
#install.packages('mlr3viz')
#install.packages('mlr3tuning')
#install.packages('GGally')

# Pakete laden
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(tidyverse)
library(mlr3tuning)

# Seed setzen
set.seed(42) 

# Daten laden
data('happy', package='GGally')
happy.original = happy
which(apply(happy.original,2,function(x) any(is.na(x))))
# zeigt an in welchen Spalten des Datensatzes NAs vorkommen
happy = happy.original %>% filter_all(all_vars(!is.na(.)))
# Filtert Datensatz dass keine NAs mehr drin sind


## Blick in die Daten
View(happy)
str(happy) 
# zeigt Struktur des Datensatzes


## Daten in Trainings- & Testdaten aufteilen
train.set.happy = sample(nrow(happy), nrow(happy)/3*2)
# Zieht zufällig 2/3 der Reihen aus happy
test.set.happy = setdiff(1:nrow(happy), train.set.happy)
# Sucht von 1 bis Anzahl an Reihen in happy, welche nicht im Trainingsset vorkommen


## Task definieren
task.happy = TaskClassif$new(id='happy', backend=happy, target='happy')
# Definiert einen neuen Klassifikationstask
## id bezeichnet den Namen des Tasks;
## backend legt die Daten fest für die der Task verwendet werden soll;
## target legt die Targetvariable fest

## Random Forest
# Learner definieren
learner.happy = mlr_learners$get('classif.ranger')
print(learner.happy)
# Parameter des Learners festlegen
learner.happy$param_set$values = list(mtry=length(task.happy$feature_names)/3, 
                                      num.trees=1000)
# mtry legt die Anzahl der Features fest die bei jedem Split zufällig gezogen werden
# num.trees legt die Anzahl der Bäume fest die erstellt werden

# Modell trainieren
model.happy = learner.happy$train(task=task.happy, row_ids=train.set.happy)
# task = gibt an mit welchem Task das Modell trainiert wird
# Mit row_ids wird festgelegt mit welchem Teil der Daten trainiert werden soll

# Vorhersagen
predictions.happy = learner.happy$predict(task.happy, row_ids=test.set.happy)
# Speichert die Vorhersagen für das Testset des Modells
print(predictions.happy)

## Evaluation
predictions.happy$confusion
# Gibt eine Confusion-matrix aus
predictions.happy$score(msr('classif.acc'))
# Gibt die Accuracy an mit der das Modell Vorhersagen macht
learner.happy$oob_error()
# Gibt den Out-of-Bag Error an

# Visualisierung
autoplot(predictions.happy)


## Parametertuning

# Parameter festlegen
params.happy = ParamSet$new(list(
  mtry = p_int(lower=1, upper=length(task.happy$feature_names)),
  min.node.size = p_int(lower=1, upper=10)))
# lower & upper legen die Unter- bzw. Obergrenze fest
# min.node.size gibt die Knotenanzahl an

# Festlegen wann Tuning beendet wird
terminator.happy = trm('evals', n_evals=10)
# n_evals legt fest wieviele Runden evaluiert werden soll

# Resampling und Evaluationsmaß festlegen
resample.happy = rsmp('cv', folds=10)
# cv = crossvalidation
# folds legt die Anzahl der Falten in der Kreuzvalidierung fest
measure.happy = msr('classif.acc')

# Tuning Instance definieren
instance.happy = TuningInstanceSingleCrit$new(
  task=task.happy,
  learner=learner.happy,
  resampling=resample.happy,
  measure=measure.happy,
  search_space=params.happy,
  terminator=terminator.happy
)
# Es werden die vorher festgelegten Parameter und Maße etc. übergeben

#Tunen
tuner.happy.random = tnr('random_search')
tuner.happy.random$optimize(instance.happy)

# Ergebnisse des Tunings auslesen
instance.happy$result_learner_param_vals

# Hyperparameter des Learners mit den Ergebnissen festlegen
learner.happy.tuned = learner.happy
learner.happy.tuned$param_set$values = instance.happy$result_learner_param_vals

# nochmal mit besseren Parametern trainieren
learner.happy.tuned$train(task.happy, row_ids=train.set.happy)

# neue Vorhersage
prediction.happy.tuned = learner.happy.tuned$predict(task.happy, row_ids=test.set.happy)
prediction.happy.tuned$score(msr('classif.acc'))

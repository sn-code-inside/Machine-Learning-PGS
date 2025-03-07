#############################
### Kapitel 5 Optimierung ###
#############################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('mlr3')
#install.packages('mlr3learners')
#install.packages('mlr3viz')
#install.packages('mlr3tuning')
#install.packages('ranger')
#install.packages('palmerpenguins')

## Pakete laden
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(palmerpenguins)

## Seed setzen
set.seed(421)

## Daten laden und inspizieren 
palmerpenguins = data.frame(penguins)
pingus.no.na = palmerpenguins[!apply(palmerpenguins,1,function(x) any(is.na(x))),]
names(pingus.no.na)

pingus = pingus.no.na[-3]


## Task definieren
task.pingus = TaskClassif$new(id = 'penguins', backend = pingus, target = 'species')
task.pingus$feature_names


## Learner definieren
learner.pingus = lrn("classif.ranger")
learner.pingus.tuned = learner.pingus


## Daten in Trainings- und Testdaten aufteilen
n = task.pingus$nrow
train.set.pingus = sample(n, size = 2/3*n)
test.set.pingus = setdiff(1:n, train.set.pingus)


## Performance ohne Tuning bestimmen
learner.pingus$train(task.pingus, row_ids = train.set.pingus)

prediction.pingus = learner.pingus$predict(task.pingus, row_ids = test.set.pingus)
prediction.pingus$confusion
prediction.pingus$score(msr('classif.acc'))


## Parameter festlegen die getuned werden sollen
params.pingus = ParamSet$new(list(
  mtry = p_int(lower = 1, upper = length(task.pingus$feature_names)),
  min.node.size = p_int(lower = 1, upper = 10)
))
params.pingus

# Festlegen wann Tuning beendet wird
terminator.pingus = trm('evals', n_evals = 50)


# Resampling und Evaluationsmaß festlegen
res.pingus = rsmp('cv', folds = 10)
measure.pingus = msr('classif.acc') 


## Tuning Instance definieren
instance.pingus = TuningInstanceSingleCrit$new(
  task = task.pingus,
  learner = learner.pingus.tuned,
  resampling = res.pingus,
  measure = measure.pingus,
  search_space = params.pingus,
  terminator = terminator.pingus
)
instance.pingus

#tuner.pingus.grid = tnr('grid_search', resolution = 2)
tuner.pingus.random = tnr('random_search')


## Tunen
#tuner.pingus.grid$optimize(instance.pingus)
tuner.pingus.random$optimize(instance.pingus)


## Ergebnisse des Tunings auslesen
instance.pingus$result_learner_param_vals

## Hyperparameter des Learners nach dem Ergebnis des Tunings einstellen
learner.pingus.tuned$param_set$values = instance.pingus$result_learner_param_vals


# nochmal mit besseren Parametern trainieren
learner.pingus.tuned$train(task.pingus, row_ids = train.set.pingus)

# neue Vorhersage
prediction.pingus.tuned = learner.pingus.tuned$predict(task.pingus, row_ids = test.set.pingus)
prediction.pingus.tuned$score(msr('classif.acc'))

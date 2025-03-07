
####################################################
### Kapitel 6 Interpretierbares Machine Learning ###
###                mit mlr3                      ###
####################################################

# falls Pakete fehlen, diese installieren
if(!require("iml", character.only = TRUE)){
  install.packages("iml")
}

# und Pakete aktivieren
library(iml)
library(mlr3)
library(mlr3learners)

# das Arbeitsverzeichnis auf den Speicherort dieses Skripts legen
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# den Datensatz laden
performance = read.csv(paste0(getwd(), '/StudentsPerformance.csv'), stringsAsFactors = TRUE)

# den Learner-Task definieren
task.reading = TaskRegr$new("Reading Performance", backend = performance, target = "reading.score")

# den Algorithmus auswählen, mit dem gelernt werden soll 
learner = lrn("regr.ranger")

# das Modell trainieren
learner$train(task.reading)

# das Prädiktorobjekt anlegen
features = mlr3misc::remove_named(performance, "reading.score")
predictor = Predictor$new(learner, data = features, y = performance$reading.score)

###########################
### Variable Importance ###
###########################

# Variable importance berechnen 
imp = FeatureImp$new(predictor, loss = "mse")

# Variable importance grafisch darstellen
imp$plot() + theme_minimal(base_size = 26)


### Indidividual conditional expectation (ICE) curves und partial dependence plots (PDP)
pdp = FeatureEffect$new(predictor, feature = "math.score", method = "pdp+ice")
pdp$plot() + theme_minimal(base_size = 26)


#######################
### Counterfactuals ###
#######################

# den Datensatz laden
performance = read.csv(paste0(getwd(), '/StudentsPerformance.csv'), stringsAsFactors = TRUE)


#  `counterfactuals` package aktiviern
library(counterfactuals)

# die interessierende Beobachtung aus dem Datensatz extrahieren
performance[duplicated(performance), ]
x.interest = performance[10,]
performance = performance[-10,]


# Prädiktionstask definieren
task.lunch = TaskClassif$new("Reading Performance", backend = performance, target = "lunch")

# das Modell trainieren, Anmerkung: wir setzen 'predict_type = "prob"', um Wahrscheinlichkeiten zu erhalten
learner = lrn("classif.ranger", predict_type = "prob")
learner$train(task.lunch)

# Prädiktor erstellen
features = mlr3misc::remove_named(performance, "lunch")
predictor = Predictor$new(learner, data = features, type = "prob", class = "standard")

# Counterfactual-Objekt anlegen
whatif = WhatIfClassif$new(predictor, n_counterfactuals = 1L)

# Modellvorhersage für interessierende Beobachtung ansehen, verwenden, um gewünschte WKt zu definieren
predictor$predict(x.interest)
cfe = whatif$find_counterfactuals(x.interest, desired_class = "standard", desired_prob = c(0.6, 1))

# Ergebnis: welche Werte müssten sich wie veröndern, damit die gewünschte Wahrscheinlichkeit erreicht wird?
data.frame(cfe$evaluate(show_diff = TRUE))



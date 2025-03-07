
####################################################
### Kapitel 6 Interpretierbares Machine Learning ###
###               mit tidymodels                 ###
####################################################

library(tidymodels)
library(vip)
library(pdp)
library(ggplot2)


# das Arbeitsverzeichnis auf den Speicherort dieses Skripts legen
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# den Datensatz laden
performance = read.csv(paste0(getwd(), '/StudentsPerformance.csv'), stringsAsFactors = TRUE)



# Datensatz in Trainingsset und Testset teilen
data_split <- initial_split(performance, prop = 0.9)

train_data <- training(data_split)
test_data <- testing(data_split)


# Modell spezifizieren
spec <- rand_forest() %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification") 

trained_model <- spec %>% 
  fit(formula = lunch ~ .,
      data = train_data)


# Modell testen (optional)
predictions <- predict(trained_model,
                       new_data = test_data,
                       type = "class")


###########################
### Variable Importance ###
###########################

# Variable Importance berechnen
var_importance <- vi(trained_model, sort = FALSE)

# Variable Importance grafisch darstellen
vip(trained_model, num_features = 13, geom = "point", horizontal = TRUE,
    aesthetics = list(color = "red", shape = 4, size = 5)) +
  theme_light()


##################
###  PDP-Plots ###
##################

# ICE-Information für ein einzelnes Feature erstellen hier: math.score
pdp_data <- partial(trained_model, pred.var = "math.score", 
                    train = train_data, center = TRUE) 

# ICE-Informationen nutzen, um ICE-Plots zu erstellen
pdp_plot <- autoplot(pdp_data) + 
  ggtitle("PDP Curves for math.score") + 
  theme_bw()

# Plot aufrufen
pdp_plot



##################
###  ICE-Plots ###
##################

# ICE-Information für ein einzelnes Feature erstellen hier: math.score
ice_data <- partial(trained_model, pred.var = "math.score", 
                    train = train_data, ice = TRUE, center = TRUE) 

# ICE-Informationen nutzen, um ICE-Plots zu erstellen
ice_plot <- autoplot(ice_data, alpha = 0.1) + 
  ggtitle("ICE Curves for math.score") + 
  theme_minimal()

# Plot aufrufen
ice_plot


#######################
### Counterfactuals ###
#######################


#  counterfactuals package und iml package aktivieren
library(counterfactuals)
library(iml)

# die interessierende Beobachtung aus dem Datensatz extrahieren - hier Beobachtung 10
performance[duplicated(performance), ]
x.interest = performance[10,]
performance = performance[-10,]


# Prädiktor mit iml package erstellen
features = performance[, - grep("lunch", names(performance))]
predictor = Predictor$new(trained_model, data = features, type = "prob", class = ".pred_standard")

# Counterfactual-Objekt anlegen
whatif = WhatIfClassif$new(predictor, n_counterfactuals = 1L)

# Modellvorhersage für interessierende Beobachtung ansehen, verwenden, um gewünschte WKt zu definieren
predictor$predict(x.interest)
cfe = whatif$find_counterfactuals(x.interest, desired_class = ".pred_standard", desired_prob = c(0.6, 1))

# Ergebnis: welche Werte müssten sich wie veröndern, damit die gewünschte Wahrscheinlichkeit erreicht wird?
data.frame(cfe$evaluate(show_diff = TRUE))







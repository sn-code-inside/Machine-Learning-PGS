#############################################
### Kapitel 6.5 Neuronale Netzwerkmodelle ###
#############################################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('tidymodels')
#install.packages('mlbench')
#install.packages('keras')

## Pakete laden
library(tidyverse)
library(tidymodels)

## Seed setzen
set.seed(42)

## Daten laden und inspizieren
data('BostonHousing', package = 'mlbench')
bh_data = tibble(BostonHousing)
print(bh_data)


## Daten in Trainings- & Testdaten aufteilen
split_bh = initial_split(bh_data, prop = 0.75)
train_set_bh = training(split_bh)
test_set_bh = testing(split_bh)


## aNN trainieren
# Modell bzw. Learner spezifizieren
ann_model = linear_reg(mode='regression') %>%
  set_engine('keras')

# Modell fitten
ann_fit = ann_model %>% 
  fit(
  formula=medv ~ .,
  data=train_set_bh
  )

# Parameter inspizieren
ann_fit %>% 
  print

# Vorhersage
ann_fit %>%
  predict(test_set_bh) %>%
  bind_cols(test_set_bh) %>%
  glimpse()


## Evaluation
# Modelle evaluieren
ann_fit %>%
  predict(test_set_bh) %>%
  bind_cols(test_set_bh) %>%
  metrics(truth=medv, estimate=.pred)

# Vorhersage visulaisieren
ann_fit %>%
  predict(test_set_bh) %>%
  bind_cols(test_set_bh) %>% 
  ggplot(aes(x=.pred, y=medv)) +
  geom_point() +
  geom_smooth(method='lm', se=F)

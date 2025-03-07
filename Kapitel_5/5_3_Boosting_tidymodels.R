############################
### Kapitel 6.3 Boosting ###
############################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('tidymodels')
#install.packages('carData')
#install.packages('xgboost')

## Pakete laden
library(tidyverse)
library(tidymodels)

## Seed setzen
set.seed(42)

## Daten laden und inspizieren
data('Freedman', package='carData')
Freedman_data = tibble(Freedman)
print(Freedman_data)

# Zeigt an in welchen Sfriedman.test()# Zeigt an in welchen Spalten des Datensatzes NAs vorkommen
Freedman_data %>% 
  summarise(across(everything(), ~ sum(is.na(.x))))

# Filtert Datensatz dass keine NAs mehr drin sind
Freedman_data %>% 
  filter_all(all_vars(!is.na(.))) -> crime


## Daten in Trainings- & Testdaten aufteilen
split_crime = initial_split(crime, prop = 0.75)
train_set_crime = training(split_crime)
test_set_crime = testing(split_crime)


## Boosted tree trainieren
# Modell bzw. Learner spezifizieren
boosting_model = boost_tree(trees=1000, mode='regression') %>%
  set_engine("xgboost")

# Modell fitten
boosting_fit = boosting_model %>% 
  fit(
  formula=crime ~ .,
  data=train_set_crime
  )

# Parameter inspizieren
boosting_fit %>% 
  print

# Vorhersage
boosting_fit %>%
  predict(test_set_crime) %>%
  bind_cols(test_set_crime) %>%
  glimpse()


## Evaluation
# Modelle evaluieren
boosting_fit %>%
  predict(test_set_crime) %>%
  bind_cols(test_set_crime) %>%
  metrics(truth=crime, estimate=.pred)

# Vorhersage visulaisieren
boosting_fit %>%
  predict(test_set_crime) %>%
  bind_cols(test_set_crime) %>% 
  ggplot(aes(x=.pred, y=crime)) +
  geom_point() +
  geom_smooth(method='lm', se=F)

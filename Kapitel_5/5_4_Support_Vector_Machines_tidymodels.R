###########################################
### Kapitel 6.4 Support Vector Machines ###
###########################################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('tidymodels')
#install.packages('carData')
#install.packages('kernlab')

## Pakete laden
library(tidyverse)
library(tidymodels)

## Seed setzen
set.seed(42)

## Daten laden und inspizieren
data('BEPS', package='carData')
BEPS_data = tibble(BEPS)
print(BEPS_data)

## Daten in Trainings- & Testdaten aufteilen
split_BEPS = initial_split(BEPS_data, prop = 0.75)
train_set_BEPS = training(split_BEPS)
test_set_BEPS = testing(split_BEPS)


## Support Vector Machine trainieren
# Modell bzw. Learner spezifizieren
svm_model = svm_linear(mode='classification') %>%
  set_engine('kernlab')

# Modell fitten
svm_fit = svm_model %>% 
  fit(
  formula=gender ~ .,
  data=train_set_BEPS
  )

# Parameter inspizieren
svm_fit %>% 
  print

# Vorhersage
svm_fit %>%
  predict(test_set_BEPS) %>%
  bind_cols(test_set_BEPS) %>%
  glimpse()


## Evaluation
# Modelle evaluieren
svm_fit %>%
  predict(test_set_BEPS) %>%
  bind_cols(test_set_BEPS) %>%
  metrics(truth=gender, estimate=.pred_class)

# Vorhersage visulaisieren
svm_fit %>%
  predict(test_set_BEPS) %>%
  bind_cols(test_set_BEPS) %>% 
  mutate(Vorhersage = if_else(.pred_class == gender, 'korrekt', 'inkorrekt')) %>% 
  ggplot(aes(x=Vorhersage, fill=Vorhersage)) +
  geom_bar()
  
  ggplot(aes(x=.pred_class, y=gender)) +
  geom_point() +
  geom_smooth(method='lm', se=F)
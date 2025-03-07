#################################
### Kapitel 6.2 Random Forest ###
#################################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('tidymodels')
#install.packages('GGally')
#install.packages('ranger')


# Pakete laden
library(tidyverse)
library(tidymodels)

# Seed setzen
set.seed(42)

# Daten laden
data('happy', package='GGally')
happy_original = happy

# Zeigt an in welchen Spalten des Datensatzes NAs vorkommen
happy %>% 
  summarise(across(everything(), ~ sum(is.na(.x))))

# Filtert Datensatz dass keine NAs mehr drin sind
# Das Feature 'id' wird entfernt, da es nichtt informativ ist
happy %>% 
  filter_all(all_vars(!is.na(.))) %>% 
  select(-id) %>% 
  tibble -> happy_nm
 

## Blick in die Daten
happy_nm %>% 
  glimpse

# zeigt Struktur des Datensatzes

## Daten in Trainings- & Testdaten aufteilen
split_happy = initial_split(happy_nm, prop=.75)
train_set_happy = training(split_happy)
test_set_happy = testing(split_happy)

## Random Forest trainieren
# Modell bzw. Learner spezifizieren
rf_model = rand_forest(trees=100, mode='classification') %>%
  set_engine('ranger')

# Modell fitten (der Punkt steht für alle Features im Datensatz)
rf_fit = rf_model %>% 
  fit(
    formula=happy ~ .,
    data=train_set_happy
  )

# Parameter inspizieren
print(rf_fit)

# Vorhersage
rf_fit %>%
  predict(test_set_happy) %>%
  bind_cols(test_set_happy) %>%
  glimpse()


## Evaluation
# Modelle evaluieren
rf_fit %>%
  predict(test_set_happy) %>%
  bind_cols(test_set_happy) %>%
  metrics(truth = happy, estimate = .pred_class)


# Vorhersage visulaisieren
rf_fit %>% 
  predict(test_set_happy) %>% 
  bind_cols(test_set_happy) %>% 
  mutate(Vorhersage = if_else(.pred_class == happy, 'korrekt', 'inkorrekt')) %>% 
  ggplot(aes(x=Vorhersage, fill=Vorhersage)) +
  geom_bar()
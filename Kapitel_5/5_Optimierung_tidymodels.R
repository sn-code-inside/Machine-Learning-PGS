#############################
### Kapitel 5 Optimierung ###
#############################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('tidymodels')
#install.packages('ranger')
#install.packages('palmerpenguins')

## Pakete laden
library(tidyverse)
library(tidymodels)

## Seed setzen
set.seed(421)

## Daten laden und inspizieren
data('penguins', package='palmerpenguins')
print(penguins)
penguins %>% 
  filter_all(all_vars(!is.na(.))) %>% 
  tibble -> penguin_data


## Daten in Trainings- & Testdaten aufteilen
split_penguins = initial_split(penguin_data, prop = 0.75)
train_set_penguins = training(split_penguins)
test_set_penguins = testing(split_penguins)


## Modell/Learner und Tuning vorbereiten
# Mögliche Tuningparameter können mit der args()-Funktion angezeigt werden
args(rand_forest)


# Tuningparameter können während der Modellspezifikation festgelegt werden
rand_forest(
  trees=1000,
  min_n = tune(),
  mtry=tune()
) %>% 
  set_engine('ranger') %>% 
  set_mode('classification') -> spec_rf

# Überprüfen der Tuningspezifikation
spec_rf


## Vorbereitung Raster für Tuning und Resampling-Vorschrift
# Erstellen eines Tuningrasters. Getunte Hyperparameter:
# Minimale Anzahl von Fällen pro Knoten
# Anzahl der Variablen für die Split Variable Randomization (Feature Sampling)
grid_rf = expand_grid(
  min_n = seq(from=4, to=8),
  mtry=seq(from=2, to=8)
)

grid_rf %>% 
  print(n=Inf)

# Erstellen einer Resampling-Beschreibung (v = 10 ist der Standardwert)
folds_rf = vfold_cv(train_set_penguins, v=10)


## Workflow für die Modellschätzung erstellen
# Wenn das Modell eine aufwendigere Datenvorverarbeitung erfordert, 
# ann auch 'add_recipe()', mit zusätzlichen Vorverarbeitungsschritten,
# anstelle von 'add_formula()' verwendet werden.
workflow() %>%
  add_model(spec_rf) %>%
  add_formula(species ~ .) -> wf_rf


# Erstellen eines Resampling-Objekts unter Verwendung des Workflows und des Tuningrasters
wf_rf %>% 
  tune_grid(
    resamples = folds_rf,
    grid = grid_rf
  ) -> res_rf

names(penguin_data)
# Die Warnungen kommen durch die Angabe, dass alle Variablen als Features verwendet werden sollen,
# also den '.', eine allerdings das Target ist (aämlich species).
res_rf

# Sammeln der Metriken für alle Hyperparameterkombinationen
res_rf %>% 
  collect_metrics() %>% 
  print(n=Inf)


# Erstellen eines Diagramms, das die mittlere Genauigkeit und die mittlere Fläche unter der ROC-Kurve 
# für alle Kombinationen der während der Abstimmung verwendeten Hyperparameter anzeigt
res_rf %>%
  collect_metrics() %>%
  mutate(mtry = factor(mtry)) %>%
  ggplot(aes(min_n, mean, color = mtry)) +
  geom_line(linewidth = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = 'free', nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = 'plasma', begin = .9, end = 0)

# Hyperparameterkombinationen mit den besten 3 Accuracies zeigen
res_rf %>%
  show_best(metric='accuracy', n=3)

# Modell mit der höchsten Accuracy auswählen
res_rf %>%
  select_best(metric='accuracy') -> best_rf

# Workflow mit den Hyperparametern des besten Modells finalisieren
wf_rf %>% 
  finalize_workflow(best_rf) -> final_wf_rf

# Finalisierten Workflow fitten
final_fit_rf = 
  final_wf_rf %>%
  last_fit(split_penguins) 

# Evaluiere den finalen Fit
final_fit_rf %>%
  collect_metrics()

# Anzahl korrekter und inkorrekter Vorhersagen visualisiseren
final_fit_rf %>%
  collect_predictions() %>% 
  mutate(Vorhersage=if_else(species == .pred_class, 'korrekt', 'inkorrekt')) %>%
  mutate(Vorhersage=factor(Vorhersage, levels=c('korrekt', 'inkorrekt'))) %>% 
  ggplot(aes(x=Vorhersage, fill=species)) + 
  geom_bar() +
  scale_x_discrete(drop=FALSE)

# Workflow einschließlich des gefitteten Modells extrahieren
final_rf = extract_workflow(final_fit_rf)
final_rf

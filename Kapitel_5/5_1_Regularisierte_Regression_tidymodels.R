#############################################
### Kapitel 6.1 Regularisierte Regression ###
#############################################

## Fehlende Pakete installieren

#install.packages('tidyverse')
#install.packages('tidymodels')
#install.packages('carData')
#install.packages('glmnet')

## Pakete laden
library(tidyverse)
library(tidymodels)

## Seed setzen
set.seed(42)

## Daten laden und inspizieren
data('Wong', package='carData')
Wong_data = tibble(Wong)
print(Wong_data)


## Daten in Trainings- & Testdaten aufteilen
split_Wong = initial_split(Wong_data, prop = 0.75)
train_set_Wong = training(split_Wong)
test_set_Wong = testing(split_Wong)


## Lasso Modell trainieren
# Modell bzw. Learner spezifizieren
lasso_model = linear_reg(penalty = 0.1, mixture = 1) %>%
  set_engine('glmnet')

# Modell fitten
lasso_fit = lasso_model %>% 
  fit(
  formula=viq ~ days + duration + sex + age + piq,
  data=train_set_Wong
  )

# Parameter inspizieren
lasso_fit %>% 
  tidy() %>% 
  print(n=Inf)

# Vorhersage
lasso_fit %>%
  predict(test_set_Wong) %>%
  bind_cols(test_set_Wong) %>%
  glimpse()


## Ridge Modell trainieren
# Modell bzw. Learner spezifizieren
ridge_model = linear_reg(penalty = 0.1, mixture = 0) %>%
  set_engine('glmnet')

# Modell fitten
ridge_fit = ridge_model %>% 
  fit(
    formula=viq ~ days + duration + sex + age + piq,
    data=train_set_Wong
  )

# Parameter inspizieren
ridge_fit %>% 
  tidy() %>% 
  print(n=Inf)

# Vorhersage
ridge_fit %>%
  predict(test_set_Wong) %>%
  bind_cols(test_set_Wong) %>%
  glimpse()


## Evaluation
# Modelle evaluieren
lasso_fit %>%
  predict(test_set_Wong) %>%
  bind_cols(test_set_Wong) %>%
  metrics(truth=viq, estimate=.pred)

ridge_fit %>%
  predict(test_set_Wong) %>%
  bind_cols(test_set_Wong) %>%
  metrics(truth=viq, estimate=.pred)

# Vorhersage visulaisieren
lasso_fit %>%
  predict(test_set_Wong) %>%
  bind_cols(test_set_Wong) %>% 
  ggplot(aes(x=.pred, y=viq)) +
  geom_point() +
  geom_smooth(method='lm', se=F)

ridge_fit %>%
  predict(test_set_Wong) %>%
  bind_cols(test_set_Wong) %>% 
  ggplot(aes(x=.pred, y=viq)) +
  geom_point() +
  geom_smooth(method='lm', se=F)


## Vergleich Modelle

lasso_fit %>%
  predict(test_set_Wong) %>%
  bind_cols(test_set_Wong) %>%
  metrics(truth=viq, estimate=.pred) %>%
  mutate(model='lasso') %>% 
  bind_rows(
    ridge_fit %>%
      predict(test_set_Wong) %>%
      bind_cols(test_set_Wong) %>%
      metrics(truth=viq, estimate=.pred) %>% 
      mutate(model='ridge')
  ) %>%
  ggplot(aes(x=.metric, y=.estimate, color=model)) +
    geom_point(size=3, alpha=.5)
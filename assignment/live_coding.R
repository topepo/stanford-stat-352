# ------------------------------------------------------------------------------
# Load some packages

library(tidymodels)
library(embed)
library(themis)
library(doMC)
# library(doParallel) # for windows users

# ------------------------------------------------------------------------------
# Set some options

tidymodels_prefer()      # to resolve typical issues like stats::filter and dplyr::filter
theme_set(theme_bw())    # ssshhhhh! don't tell Hadley :-)
registerDoMC(cores = 10) # see https://www.tmwr.org/resampling.html#parallel
# Note about intel vs apple silicon

# ------------------------------------------------------------------------------

load("assignment/hotel_stays.RData")

# we have an hour so maybe sample down the data for speed?
set.seed(1)
hotel_stays <- hotel_stays %>% sample_n(10000)

# ------------------------------------------------------------------------------
# Data splitting before anything else!
# We'll use a validation set. See https://www.tmwr.org/resampling.html#validation

set.seed(2)
hotel_split <- initial_split(hotel_stays, strata = children)

hotel_train <- training(hotel_split)
hotel_test <- testing(hotel_split)

# ------------------------------------------------------------------------------
# Let's look at the data for a bit

str(hotel_train)
hotel_train %>% View()

hotel_train %>% count(children)

hotel_train %>%
  group_by(booking_agent) %>%
  summarize(prop = mean(children == "yes"), n = n(), .groups = "drop") %>%
  ggplot(aes(x = prop, y = reorder(booking_agent, prop), fill = log10(n + 1))) +
  geom_bar(stat = "identity")

hotel_train %>%
  ggplot(aes(lead_time)) +
  geom_histogram()

# ------------------------------------------------------------------------------
# Get some data preprocessing together

basic_rec <-
  recipe(children ~ ., data = hotel_train) %>%
  step_mutate(lead_time = log(lead_time + 1)) %>%
  step_lencode_mixed(booking_agent, country, outcome = vars(children))


ind_rec <-
  basic_rec %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ------------------------------------------------------------------------------
# resampling

set.seed(3)
rs <- validation_split(hotel_train)


# ------------------------------------------------------------------------------
# How to measure performance?

cls_metrics <- metric_set(roc_auc, mn_log_loss, sensitivity, specificity)

# ------------------------------------------------------------------------------
# ancillary objects

grd_ctrl <- control_grid(save_pred = TRUE)

# ------------------------------------------------------------------------------
# Fit a few models

glm_res <-
  logistic_reg() %>%
  fit_resamples(ind_rec, resamples = rs, metrics = cls_metrics, control = grd_ctrl)

collect_metrics(glm_res)

glm_res %>%
  collect_predictions() %>%
  roc_curve(children, .pred_yes) %>%
  autoplot()

set.seed(4)
glmnet_res <-
  logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  tune_grid(ind_rec, resamples = rs, metrics = cls_metrics,
            control = grd_ctrl, grid = 20)

best_glmnet <- select_best(glmnet_res, metric = "roc_auc")
best_glmnet <- tibble(penalty = 0.004, mixture = 1)

set.seed(5)
rf_res <-
  rand_forest(mtry = tune(), trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")  %>%
  tune_grid(basic_rec, resamples = rs, metrics = cls_metrics,
            control = grd_ctrl, grid = 20)


# ran out of time here :-(

# ------------------------------------------------------------------------------
# if there is time: model explainability


# ------------------------------------------------------------------------------
# evaluate the test set





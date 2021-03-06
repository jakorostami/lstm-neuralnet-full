###Ladda dessa paket
library(tidyverse)
library(glue)
library(forcats)

# Time Series
library(timetk)
library(tidyquant)
library(tibbletime)

# Visualization
library(cowplot)

# Preprocessing
library(recipes)

# Sampling / Accuracy
library(rsample)
library(yardstick) 

# Modeling
library(keras)
library(tfruns)
library(tensorflow)

use_session_with_seed(2019)
####################################################
##Datamaterialet �r fr�n Svensk M�klarstatistik och delas INTE med utan deras tillst�nd
#N�r du har l�st in datan (endast medelpriser), formatera d� om till en tidsserie med koden nedan

sthlmvilla <- read.table("clipboard", header=TRUE, sep="\t", dec=",")
tidsserie <- ts(sthlmvilla, start=c(2005,1), end=c(2019,3), frequency=12)


##G�r om till en tibble med tidsseriens index som as_date
sthlmvilla <- tidsserie %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index = index)


###Plotta i en graf med tv� figurer
p1 <- sthlmvilla %>% 
  filter_time("2005"~"end") %>%
  ggplot(aes(index, value)) +
  geom_point(color = palette_light()[[6]], alpha = 0.4) +
  geom_line(color = palette_light()[[1]], alpha = 0.9) +
  theme_tq() +
  labs(
    title = "Medelpris/m�n Stor-G�teborg - Bostadsr�tt"
  )+xlab(label="Tidsperiod")+ylab(label="SEK")

p2 <- sthlmvilla %>%
  filter_time("2011" ~ "end") %>%
  ggplot(aes(index, value)) +
  geom_line(color = palette_light()[[6]], alpha = 0.5) +
  geom_point(color = palette_light()[[1]]) +
  theme_tq() +
  labs(
    title = "Medelpris/m�n Stor-G�teborg - Bostadsr�tt (Inzoomad f�r att visa cykliskt)",
    caption = "K�lla: Svensk M�klarstatistik"
  )


p_title <- ggdraw() + 
  draw_label("Bostadsr�tter Stor-G�teborg", size = 18, fontface = "bold", colour = palette_light()[[1]])

###Plotta tv� grafer
plot_grid(p_title, p1, p2, ncol = 1, rel_heights = c(0.1, 1, 1))


####Dags f�r ACF visualisering
tidy_acf <- function(data, value, lags = 0:20) {
  
  value_expr <- enquo(value)
  
  acf_values <- data %>%
    pull(value) %>%
    acf(lag.max = tail(lags, 1), plot = FALSE) %>%
    .$acf %>%
    .[,,1]
  
  ret <- tibble(acf = acf_values) %>%
    rowid_to_column(var = "lag") %>%
    mutate(lag = lag - 1) %>%
    filter(lag %in% lags)
  
  return(ret)
}


#####S�tt max lag 15 �r tillbaka
max_lag <- 12 * 14

###Tibble med ACF
sthlmvilla %>%
  tidy_acf(value, lags = 0:max_lag)

####Jag har satt lag=12 i denna graf
sthlmvilla %>%
  tidy_acf(value, lags = 0:max_lag) %>%
  ggplot(aes(lag, acf)) +
  geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
  geom_vline(xintercept = 12, size = 1, color = palette_light()[[2]]) +
  annotate("text", label = "12 Month Mark", x = 13, y = 0.8, 
           color = palette_light()[[2]], size = 3, hjust = 0) +
  theme_tq() +
  labs(title = "ACF: Houses Metro-Stockholm")

###H�r ser vi plot p� hur det �kar vid lag=12
sthlmvilla %>%
  tidy_acf(value, lags = 1:24) %>%
  ggplot(aes(lag, acf)) +
  geom_vline(xintercept = 12, size = 2, color = palette_light()[[2]]) +
  geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
  geom_point(color = palette_light()[[1]], size = 2) +
  geom_label(aes(label = acf %>% round(2)), vjust = -1,
             color = palette_light()[[1]]) +
  annotate("text", label = "12 Month Mark", x = 13, y = 0.9, 
           color = palette_light()[[2]], size = 5, hjust = 0) +
  theme_tq() +
  labs(title = "ACF: Houses Uppsala",
       subtitle = "Zoomed in on Lags 1 to 24")

##Optimal lag funktion
optimal_lag_conf <- sthlmvilla %>%
  tidy_acf(value, lags = 10:36) %>%
  filter(acf == max(acf)) %>%
  pull(lag)

##Printa optimal lag
optimal_lag_conf
#############################################################

##H�r kan du leka med siffrorna
periods_train <- 12 *4
periods_test  <- 12*1
skip_span     <- 12*2

rolling_origin_resamples <- rolling_origin(
  sthlmvilla,
  initial    = periods_train,
  assess     = periods_test,
  cumulative = FALSE,
  skip       = skip_span
)

rolling_origin_resamples




# Plot funktion f�r enskild split
plot_split <- function(split, expand_y_axis = TRUE, alpha = 1, size = 1, base_size = 14) {
  
  # Data manipulering
  train_tbl <- training(split) %>%
    add_column(key = "training") 
  
  test_tbl  <- testing(split) %>%
    add_column(key = "testing") 
  
  data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
    as_tbl_time(index = index) %>%
    mutate(key = fct_relevel(key, "training", "testing"))
  
  # Samla dess egenskaper
  train_time_summary <- train_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  
  test_time_summary <- test_tbl %>%
    tk_index() %>%
    tk_get_timeseries_summary()
  
  # Visualisera
  g <- data_manipulated %>%
    ggplot(aes(x = index, y = value, color = key)) +
    geom_line(size = size, alpha = alpha) +
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    labs(
      title    = glue("Split: {split$id}"),
      subtitle = glue("{train_time_summary$start} to {test_time_summary$end}"),
      y = "", x = ""
    ) +
    theme(legend.position = "none") 
  
  if (expand_y_axis) {
    
    priser_time_summary <- sthlmvilla %>% 
      tk_index() %>% 
      tk_get_timeseries_summary()
    
    g <- g +
      scale_x_date(limits = c(priser_time_summary$start, 
                              priser_time_summary$end))
  }
  
  return(g)
}




rolling_origin_resamples$splits[[1]] %>%
  plot_split(expand_y_axis = TRUE) +
  theme(legend.position = "bottom")


#########################################



# Plot funktion som tar med alla splits
plot_sampling_plan <- function(sampling_tbl, expand_y_axis = TRUE, 
                               ncol = 3, alpha = 1, size = 1, base_size = 14, 
                               title = "CV-Sliding Window plan") {
  
  # Mappa plot_split() till sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map(splits, plot_split, 
                          expand_y_axis = expand_y_axis,
                          alpha = alpha, base_size = base_size))
  
  # Skapa plots med cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots 
  
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  p_title <- ggdraw() + 
    draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[3]])
  
  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
  
  return(g)
  
}


rolling_origin_resamples %>%
  plot_sampling_plan(expand_y_axis = T, ncol = 3, alpha = 1, size = 1, base_size = 10, 
                     title = "CV Strategi: Sliding Windows Stickprovs Plan")



rolling_origin_resamples %>%
  plot_sampling_plan(expand_y_axis = F, ncol = 3, alpha = 1, size = 1, base_size = 10, 
                     title = "CV Strategi: Zoomed In")



split    <- rolling_origin_resamples$splits[[4]]
split_id <- rolling_origin_resamples$id[[4]]



plot_split(split, expand_y_axis = FALSE, size = 0.5) +
  theme(legend.position = "bottom") +
  ggtitle(glue("Split: {split_id}"))

pred1 <- sample_predictions_lstm_tbl$predict[[4]]
pred1_id <- sample_predictions_lstm_tbl$id[[4]]

plot_split(pred1, expand_y_axis = FALSE, size = 0.5) +
  theme(legend.position = "bottom") +
  ggtitle(glue("Split: {pred1_id}"))


df_trn <- training(split)
df_tst <- testing(split)

df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_tst %>% add_column(key = "testing")
) %>% 
  as_tbl_time(index = index)

df



recept_obj <- recipe(value ~ ., df) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

df_processed_tbl <- bake(recept_obj, df)

df_processed_tbl

center_history <- brei$steps[[2]]$means["value"]
scale_history  <- brei$steps[[3]]$sds["value"]

center_history <- rec_obj$steps[[2]]$means["value"]
scale_history  <- rec_obj$steps[[3]]$sds["value"]

c("center" = center_history, "scale" = scale_history)




####Modeltest av delade urval (du kan skippa detta fram till l�ngst ned f�r att testa hela modellen)


# Model inputs
lag_setting  <- 12 # = nrow(df_tst)
batch_size   <- 1  
train_length <- 171
tsteps       <- 1
epochs       <- 200

######

# Tr�ningsset
lag_train_tbl <- df_processed_tbl %>%
  mutate(value_lag = lag(value, n = lag_setting)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "training") %>%
  tail(train_length)

x_train_vec <- lag_train_tbl$value_lag
x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))

y_train_vec <- lag_train_tbl$value
y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))

# Testset
lag_test_tbl <- df_processed_tbl %>%
  mutate(
    value_lag = lag(value, n = lag_setting)
  ) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "testing")

x_test_vec <- lag_test_tbl$value_lag
x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))

y_test_vec <- lag_test_tbl$value
y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), 1))


###Model fit 




model <- keras_model_sequential()

model %>%
  layer_lstm(units            = 24, 
             input_shape      = c(tsteps, 1), activation="tanh", recurrent_activation = "sigmoid",
             batch_size       = batch_size, recurrent_dropout = 0.4, 
             return_sequences = TRUE, 
             stateful         = TRUE) %>%
  layer_lstm(units            = 12, 
             input_shape      = c(tsteps, 1), activation="tanh", recurrent_activation = "sigmoid",
             batch_size       = batch_size, recurrent_dropout = 0.5, 
             return_sequences = FALSE, 
             stateful         = TRUE) %>% 
  layer_dense(units = 1, activation="linear")

model %>% 
  compile(loss = 'mae', optimizer = 'adadelta',metrics=list('mse'))

model

cero <- list(callback_early_stopping(monitor="val_loss", 
                                     patience=50, baseline=0.2,
                                     verbose=1, mode=c("auto", "min","max"),
                                     restore_best_weights = FALSE))
###FITTING

for (i in 1:epochs) {
  model %>% fit(x          = x_train_arr, 
                y          = y_train_arr, 
                batch_size = batch_size,
                epochs     = 1, 
                verbose    = 1, 
                shuffle    = FALSE)
  
  model %>% reset_states()
  cat("Epoch: ", i)
  
}

# Skapa forecasts
pred_out <- model %>% 
  predict(x_test_arr, batch_size = batch_size) %>%
  .[,1] 

# �tertransformera v�rden
pred_tbl <- tibble(
  index   = lag_test_tbl$index,
  value   = (pred_out * scale_history + center_history)^2
) 

# Kombinera faktisk data med forecasts
tbl_1 <- df_trn %>%
  add_column(key = "actual")

tbl_2 <- df_tst %>%
  add_column(key = "actual")

tbl_3 <- pred_tbl %>%
  add_column(key = "predict")

shuu <- shuu %>% 
  add_column(key="actual")


# Skapa time_bind_rows() f�r att l�sa dplyr problem
time_bind_rows <- function(data_1, data_2, index) {
  index_expr <- enquo(index)
  bind_rows(data_1, data_2) %>%
    as_tbl_time(index = !! index_expr)
}

ret <- list(tbl_1,tbl_2, tbl_3) %>%
  reduce(time_bind_rows, index = index) %>%
  arrange(key, index) %>%
  mutate(key = as_factor(key))

ret



calc_rmse <- function(prediction_tbl) {
  
  rmse_calculation <- function(data) {
    data %>%
      spread(key = key, value = value) %>%
      select(-index) %>%
      filter(!is.na(predict)) %>%
      rename(
        truth    = actual,
        estimate = predict
      ) %>%
      rmse(truth, estimate) %>% .$.estimate
  }
  
  safe_rmse <- possibly(rmse_calculation, otherwise = NA)
  
  safe_rmse(prediction_tbl)
  
}

calc_rmse(ret)

###Visualisera RMSE



# Skapa enskild plot funktion
plot_prediction <- function(data, id, alpha = 1, size = 2, base_size = 14) {
  
  rmse_val <- calc_rmse(data)
  
  g <- data %>%
    ggplot(aes(index, value, color = key)) +
    geom_line(alpha = alpha, size = size) + 
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    theme(legend.position = "none") +
    labs(
      title = glue("{id}, RMSE: {round(rmse_val, digits = 1)}"),
      x = "", y = ""
    )
  
  return(g)
}



ret %>% 
  plot_prediction(id = "Model fit", alpha = 0.65) +
  theme(legend.position = "bottom")


##########################################################################
## B�rja tr�na din modell med full funktion nedan


predict_keras_lstm <- function(split, epochs = 300, ...) {
  
  lstm_prediction <- function(split, epochs, ...) {
    
    #  Data konfiguration
    df_trn <- training(split)
    df_tst <- testing(split)
    
    df <- bind_rows(
      df_trn %>% add_column(key = "training"),
      df_tst %>% add_column(key = "testing")
    ) %>% 
      as_tbl_time(index = index)
    
    #  Preprocessing
    rec_obj <- recipe(value ~ ., df) %>%
      step_sqrt(value) %>%
      step_center(value) %>%
      step_scale(value) %>%
      prep()
    
    df_processed_tbl <- bake(rec_obj, df)
    
    center_history <- rec_obj$steps[[2]]$means["value"]
    scale_history  <- rec_obj$steps[[3]]$sds["value"]
    
    #  LSTM uppl�gg
    lag_setting  <- 12 # = nrow(df_tst)
    batch_size   <- 1
    train_length <- 171
    tsteps       <- 1
    epochs       <- epochs
    
    #  Tr�ning/test
    lag_train_tbl <- df_processed_tbl %>%
      mutate(value_lag = lag(value, n = lag_setting)) %>%
      filter(!is.na(value_lag)) %>%
      filter(key == "training") %>%
      tail(train_length)
    
    x_train_vec <- lag_train_tbl$value_lag
    x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))
    
    y_train_vec <- lag_train_tbl$value
    y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))
    
    lag_test_tbl <- df_processed_tbl %>%
      mutate(
        value_lag = lag(value, n = lag_setting)
      ) %>%
      filter(!is.na(value_lag)) %>%
      filter(key == "testing")
    
    x_test_vec <- lag_test_tbl$value_lag
    x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))
    
    y_test_vec <- lag_test_tbl$value
    y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), 1))
    
    #  LSTM Model
    model <- keras_model_sequential()
    
    model %>%
      layer_lstm(units            = 12, 
                 input_shape      = c(tsteps, 1), activation="tanh", recurrent_activation = "sigmoid",
                 batch_size       = batch_size, recurrent_dropout= 0.1,
                 return_sequences = TRUE, 
                 stateful         = TRUE) %>%
      layer_lstm(units            = 12, 
                 input_shape      = c(tsteps, 1), activation="tanh", recurrent_activation = "sigmoid",
                 batch_size       = batch_size, recurrent_dropout = 0.4, 
                 return_sequences = FALSE, 
                 stateful         = TRUE) %>% 
      
      layer_dense(units = 1, activation="linear")
    
    model %>% 
      compile(loss = 'mae', optimizer = 'adadelta', metrics=list('mse'))
    
    #  Fitting LSTM
    for (i in 1:epochs) {
      model %>% fit(x          = x_train_arr, 
                    y          = y_train_arr, 
                    batch_size = batch_size,
                    epochs     = 1, 
                    verbose    = 1,
                    shuffle    = FALSE
      )
      
      model %>% reset_states()
      cat("Epoch: ", i)
      
    }
    
    #  Forecast och returna tidy data
    # Utf�r forecasts
    pred_out <- model %>% 
      predict(x_test_arr, batch_size = batch_size) %>%
      .[,1] 
    
    # �tertransformera v�rden
    pred_tbl <- tibble(
      index   = (lag_test_tbl$index %>% tail(lag_setting)),
      value   = (pred_out * scale_history + center_history)^2
    ) 
    
    # Kombinera faktiskt data med forecasts
    tbl_1 <- df_trn %>%
      add_column(key = "actual")
    
    tbl_2 <- df_tst %>%
      add_column(key = "actual")
    
    tbl_3 <- pred_tbl %>%
      add_column(key = "predict")
    
    # Skapa time_bind_rows() f�r att l�sa dplyr problem
    time_bind_rows <- function(data_1, data_2, index) {
      index_expr <- enquo(index)
      bind_rows(data_1, data_2) %>%
        as_tbl_time(index = !! index_expr)
    }
    
    ret <- list(tbl_1, tbl_2, tbl_3) %>%
      reduce(time_bind_rows, index = index) %>%
      arrange(key, index) %>%
      mutate(key = as_factor(key))
    
    return(ret)
    
  }
  
  safe_lstm <- possibly(lstm_prediction, otherwise = NA)
  
  safe_lstm(split, epochs, ...)
  
}



predict_keras_lstm(split, epochs = 1)



sample_predictions_lstm_tbl <- rolling_origin_resamples %>%
  mutate(predict = map(splits, predict_keras_lstm, epochs = 50))



sample_predictions_lstm_tbl



## Summera stickprovs RMSE f�r dina slice

sample_rmse_tbl <- sample_predictions_lstm_tbl %>%
  mutate(rmse = map_dbl(predict, calc_rmse)) %>%
  select(id, rmse)

sample_rmse_tbl

#####

#####

sample_rmse_tbl %>%
  ggplot(aes(rmse)) +
  geom_histogram(aes(y = ..density..), fill = palette_light()[[10]], bins = 16) +
  geom_density(fill = palette_light()[[3]], alpha = 0.5) +
  theme_tq() +
  ggtitle("Histogram of RMSE")



sample_rmse_tbl %>%
  summarize(
    mean_rmse = mean(rmse),
    sd_rmse   = sd(rmse)
  )




plot_predictions <- function(sampling_tbl, predictions_col, 
                             ncol = 3, alpha = 1, size = 2, base_size = 14,
                             title = "Backtested Predictions") {
  
  predictions_col_expr <- enquo(predictions_col)
  
  
  # Mappa plot_split() till sampling_tbl
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map2(!! predictions_col_expr, id, 
                           .f        = plot_prediction, 
                           alpha     = alpha, 
                           size      = size, 
                           base_size = base_size)) 
  
  # Skapa plots med cowplot
  plot_list <- sampling_tbl_with_plots$gg_plots 
  
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  
  
  
  p_title <- ggdraw() + 
    draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
  
  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
  
  return(g)
  
}



sample_predictions_lstm_tbl %>%
  plot_predictions(predictions_col = predict, alpha = 0.5, size = 1, base_size = 10,
                   title = "Houses Metro-Stockholm: Sliding Windows Predictions")
#####################################################################################

#####################################Kod f�r f�rdig modell###############################
########################################################################################







###Forecasta kommande 12 m�nader som �r g�mda fr�n modellen
set.seed(900524)

predict_keras_lstm_future <- function(data, epochs = 300, ...) {
  
  lstm_prediction <- function(data, epochs, ...) {
    
    # Data konfig - modifierad
    df <- data
    
    # Preprocessing
    rec_obj <- recipe(value ~ ., df) %>%
      step_sqrt(value) %>%
      step_center(value) %>%
      step_scale(value) %>%
      prep()
    
    df_processed_tbl <- bake(rec_obj, df)
    
    center_history <- rec_obj$steps[[2]]$means["value"]
    scale_history  <- rec_obj$steps[[3]]$sds["value"]
    
    # LSTM plan
    lag_setting  <- 12 # = nrow(df_tst)
    batch_size   <- 1
    train_length <- 171
    tsteps       <- 1
    epochs       <- epochs
    
    # Tr�ningskonfig - modifierad
    lag_train_tbl <- df_processed_tbl %>%
      mutate(value_lag = lag(value, n = lag_setting)) %>%
      filter(!is.na(value_lag)) %>%
      tail(train_length)
    
    x_train_vec <- lag_train_tbl$value_lag
    x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))
    
    y_train_vec <- lag_train_tbl$value
    y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))
    
    x_test_vec <- y_train_vec %>% tail(lag_setting)
    x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))
    
    # LSTM model
    model <- keras_model_sequential()
    
    model %>%
      layer_lstm(units            = 24, 
                 input_shape      = c(tsteps, 1), activation="tanh", recurrent_activation = "sigmoid",
                 batch_size       = batch_size, recurrent_dropout= 0.1,
                 return_sequences = TRUE, 
                 stateful         = TRUE) %>% 
      layer_lstm(units            = 12, 
                 input_shape      = c(tsteps, 1), activation="tanh", recurrent_activation = "sigmoid",
                 batch_size       = batch_size, recurrent_dropout= 0.5,
                 return_sequences = FALSE, 
                 stateful         = TRUE) %>% 
      layer_dense(units = 1, activation="linear")
    
    model %>% 
      compile(loss = 'mae', optimizer = 'adadelta', metrics=list('mse'))
    
    # Fitting LSTM
    for (i in 1:epochs) {
      model %>% fit(x          = x_train_arr, 
                    y          = y_train_arr, 
                    batch_size = batch_size,
                    epochs     = 1, 
                    verbose    = 1
      )
      
      model %>% reset_states()
      cat("Epoch: ", i)
      
    }
    
    #  Forecast och returna tidy data
    # Utf�r forecasts
    pred_out <- model %>% 
      predict(x_test_arr, batch_size = batch_size) %>%
      .[,1] 
    
    
    # Skapa framtida index med tk_make_future_timeseries()
    idx <- data %>%
      tk_index() %>%
      tk_make_future_timeseries(n_future = lag_setting)
    
    # �tertransformera v�rden
    pred_tbl <- tibble(
      index   = idx,
      value   = (pred_out * scale_history + center_history)^2
    )
    
    # Kombinera faktisk data med forecasts
    tbl_1 <- df %>%
      add_column(key = "actual")
    
    tbl_3 <- pred_tbl %>%
      add_column(key = "predict")
    
    # Skapa time_bind_rows() f�r att l�sa dplyr problem
    time_bind_rows <- function(data_1, data_2, index) {
      index_expr <- enquo(index)
      bind_rows(data_1, data_2) %>%
        as_tbl_time(index = !! index_expr)
    }
    
    ret <- list(tbl_1, tbl_3) %>%
      reduce(time_bind_rows, index = index) %>%
      arrange(key, index) %>%
      mutate(key = as_factor(key))
    
    return(ret)
    
  }
  
  safe_lstm <- possibly(lstm_prediction, otherwise = NA)
  
  safe_lstm(data, epochs, ...)
  
}



framtida_priser_tbl <- predict_keras_lstm_future(sthlmvilla, epochs = 2)



klar_lstm_modell <- framtida_priser_tbl %>%
  filter_time("2010" ~ "end") %>%
  plot_prediction(id = NULL, alpha = 0.4, size = 1.5) + 
  theme(legend.position = "bottom") +
  ggtitle("House Prices Metro-Stockholm: 12 Month Forecast", subtitle = "Forecast Horizon: 2018-04-01 to 2019-03-01")


klar_lstm_modell

###SLUT###
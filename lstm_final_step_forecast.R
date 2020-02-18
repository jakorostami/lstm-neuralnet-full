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

#####################################Kod för färdig modell###############################
########################################################################################



###Forecasta kommande 12 månader som är gömda från modellen
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
    
    # Träningskonfig - modifierad
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
    # Utför forecasts
    pred_out <- model %>% 
      predict(x_test_arr, batch_size = batch_size) %>%
      .[,1] 
    
    
    # Skapa framtida index med tk_make_future_timeseries()
    idx <- data %>%
      tk_index() %>%
      tk_make_future_timeseries(n_future = lag_setting)
    
    # Återtransformera värden
    pred_tbl <- tibble(
      index   = idx,
      value   = (pred_out * scale_history + center_history)^2
    )
    
    # Kombinera faktisk data med forecasts
    tbl_1 <- df %>%
      add_column(key = "actual")
    
    tbl_3 <- pred_tbl %>%
      add_column(key = "predict")
    
    # Skapa time_bind_rows() för att lösa dplyr problem
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
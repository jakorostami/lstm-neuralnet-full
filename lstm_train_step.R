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

##########################################################################
## Börja träna din modell med full funktion nedan


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
    
    #  LSTM upplägg
    lag_setting  <- 12 # = nrow(df_tst)
    batch_size   <- 1
    train_length <- 171
    tsteps       <- 1
    epochs       <- epochs
    
    #  Träning/test
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
    # Utför forecasts
    pred_out <- model %>% 
      predict(x_test_arr, batch_size = batch_size) %>%
      .[,1] 
    
    # Återtransformera värden
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
    
    # Skapa time_bind_rows() för att lösa dplyr problem
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



## Summera stickprovs RMSE för dina slice

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
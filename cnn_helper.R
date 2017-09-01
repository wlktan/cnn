library(dplyr)
library(data.table)
library(MASS)
library(ggplot2)

#source("validation_bias_correction.R")

AddError <- function(train.y,
                     mu0, Sigma0,
                     mu1, Sigma1){
  # Add misclassification error to a vector of gold standard labels
  # where error is determined by sens/spec of the silver standard
  # Arguments:
  #  @gold.standard: vector of 0/1 gold standard labels
  # Outputs:
  #  @silver.standard: vector of 0/1 silver standard labels
  
  # Generate features
  p <- length(mu0)
  X <- matrix(data = NA, nrow = length(train.y), ncol = p)
  for(i in 1:dim(X)[1]){
    if(train.y[i] == 1) X[i,] <- mvrnorm(1,mu1,Sigma1)
    if(train.y[i] == 0) X[i,] <- mvrnorm(1,mu0,Sigma0)
  }
  
  # predicted probabilities
  pred.model <- glm(train.y~X, family = "binomial")
  pred.probs <- predict(pred.model, type = "response")
  pred.labels <- ifelse(pred.probs >= 0.5, 1, 0) # Bayes error
  
  error <- CalcMetrics(pred.labels,train.y)$metrics.list
  
  return(list(labels = pred.labels,
              probs = pred.probs,
              error = error))
}

GetTrainTest <- function(train.all,
                         test,
                         n_train = 1000,
                         error = FALSE){
  
  # Set up train and test datasets
  train <- train.all %>%
    sample_n(n_train) # Downsample
  
  train.x <- t(train[, -1])
  train.y <- train[, 1]
  train.array <- train.x
  dim(train.array) <- c(img_resolution, img_resolution, 1, ncol(train.x))
  
  test.x <- t(test[,-1])
  test.y <- test[, 1]
  test.array <- test.x
  dim(test.array) <- c(img_resolution, img_resolution, 1, ncol(test.x))
  
  if(error){
    p <- 30
    mu0 <- runif(p, min = 0.1, max = 0.2)
    mu1 <- runif(p, min = 0.3, max = 0.5)
    Sigma0 <- diag(p)
    Sigma1 <- diag(p)
    train.y <- AddError(train.y,mu0,Sigma0,mu1,Sigma1)$labels
  }
  
  return(list(train.array = train.array,
              train.y = train.y,
              test.array = test.array,
              test.y = test.y))
}

RunOneCNN <- function(deep.NN,
                        devices,
                        train.all,
                        test,
                        n_train = 1000,
                      error = FALSE,
                      num_rounds){
  # Runs a Feedforward neural net model with 
  # Arguments:
  #  @deep.NN: architecture of the neural network
  #  @devices: device used to perform training
  #  @train.array: tensor of training features with dimensions (height,width,depth,num_obs)
  #  @train.y: vector of training labels
  #  @test.array: tensor of testing features with dimensions (height,width,depth,num_obs)
  #  @test.y: vector of testing labels
  # Outputs:
  #  @List of:
  
  data <- GetTrainTest(train.all, test, n_train, error)
  train.array <- data$train.array
  train.y <- data$train.y
  test.array <- data$test.array
  test.y <- data$test.y
  
  # Train model
  mx.set.seed(100)
  model <- mx.model.FeedForward.create(deep.NN,
                                       X = train.array,
                                       y = train.y,
                                       ctx = devices,
                                       num.round = num_rounds,
                                       array.batch.size = 100,
                                       learning.rate = 0.02, 
                                       momentum = 0.9, 
                                       wd = 0.00001,
                                       eval.metric = mx.metric.accuracy,
                                       epoch.end.callback = mx.callback.log.train.metric(100))
  
  # Test model
  predicted.prob <- predict(model, test.array)[2,]
  predicted.labels <- predict(model, test.array) %>% # Bayes error: yhat=I(phat>0.5)
    t(.) %>%
    max.col(.) - 1
  pred <- cbind(predicted.prob = predicted.prob,
                predicted.labels = predicted.labels)
  
  test.accuracy <- table(test.y, predicted.labels) %>%
    diag(.) %>%
    sum(.)/length(test.y)
  
  return(list(model = model,
              pred = pred,
              test.accuracy = test.accuracy))
}

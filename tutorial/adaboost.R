library(rpart)

train.adaboost <- function(data, n_estimators=50){
  alphas <- array(0, n_estimators)
  estimators <- list()
  sample_weights <- array(1, nrow(data)) / nrow(data)
  y_changed <- data$label
  y_changed[data$label == 0] <- -1
  to_train <- data
  to_train$label <- y_changed
  to_train$label <- as.factor(to_train$label)
  for (iboost in seq(n_estimators)){
    estimator <- rpart(label~., to_train, sample_weights, control = rpart.control(maxdepth = 1))
    estimators[[iboost]] <- estimator
    predictions <- numeric_from_factor(predict(estimator, newdata=to_train, type = "class"))
    incorrect <- predictions != to_train$label
    error <- sum(sample_weights * incorrect)
    if (error > 0){
      alphas[iboost] <- 0.5 * log((1-error) / error)
      a <- alphas[iboost]
      modifier <- exp(-y_changed * a * predictions)
      sample_weights <- sample_weights * modifier
      sample_weights <- sample_weights / sum(sample_weights)
    }else{
      alphas[iboost] <- 1.0
      alphas <- alphas[1:iboost]
      n_estimators <- length(estimators)
      break
    }
  }
  result <- list('estimators'=estimators, 'alphas'=alphas)
  return(result)
}

predict.adaboost <- function(data, classifier){
  predictions <- array(0, nrow(data))
  estimators <- classifier$estimators
  alphas <- classifier$alpha
  for (iboost in seq(length(estimators))){
    a <- alphas[iboost]
    pred <- numeric_from_factor(predict(estimators[[iboost]], newdata=data, type = "class"))
    predictions = predictions + a * pred
  }
  probas <- 1.0 / (1.0 + exp(-2*predictions))
  result <- matrix(0,length(probas),2)
  result[,1] <- 1 - probas
  result[,2] <- probas
  return(result)
}

numeric_from_factor <- function (inputFactor){
  return(as.numeric(levels(inputFactor)[as.integer(inputFactor)]))
}

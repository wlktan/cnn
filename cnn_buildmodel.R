rm(list=ls())
library(mxnet)

source("~/Desktop/Dissertation/Surrogate sampling/Rcode/latest_code/sampling_helper.R")
source("cnn_helper.R")

# Load train and test datasets; labels are 1st col
mnist <- fread("mnist_train.csv") %>%
  rbind(., fread("mnist_test.csv")) %>%
  filter(V1 %in% c(4,7))

n_train <- floor(0.8 * nrow(mnist))
set.seed(2018)
train.id <- sample(seq_len(nrow(mnist)), size = n_train)

train.all <- mnist[train.id, ]
test <- mnist[-train.id, ]

img_resolution <- floor(sqrt(dim(train.all)[2]-1))

num_classes <- unique(test[,1]) %>%
  length(.)
print(num_classes)

# Convolutional Neural Network (LeNet architecture) -----------------------------------

# original dim: hxwxd = 28x28x1
deep.NN <- mx.symbol.Convolution(data = mx.symbol.Variable('data'), # project to lower resolution but increase depth
                                 kernel = c(5,5), 
                                 stride = c(1,1),
                                 num.filter = 5) %>% # 24x24x5
  mx.symbol.Activation(data = ., # apply a non-linear function
                       act_type = "sigmoid") %>%
  mx.symbol.Pooling(data = .,
                    pool_type = "max",
                    kernel = c(2,2), 
                    stride = c(2,2))  %>%# 12x12x5
  mx.symbol.Flatten(data =.) %>%
  mx.symbol.FullyConnected(data = ., 
                           num_hidden = 120) %>% # 120x1
  mx.symbol.Activation(data = ., 
                       act_type = "sigmoid") %>%
  mx.symbol.FullyConnected(data = .,
                           num_hidden = 2) %>% # 2x1
  mx.symbol.SoftmaxOutput(data = .)

# Run CNN ----------------------------------------------------------------
train.size.grid <- c(50,75,100,150,200,250,300,350,400,450,500,550,600,700,800)

devices <- mx.cpu()

# Base CNN model (no misclassification)
model1.accuracy <- apply(as.data.frame(train.size.grid), 1, function(x)
  RunOneCNN(deep.NN,devices,train.all,test,n_train = x,num_rounds = 20)$test.accuracy)

plot(train.size.grid, model1.accuracy, type = "l",
     main = "Learning Curve", 
     xlab = "training size", 
     ylab = "accuracy")

# Labels with misclassification
model2.accuracy <- apply(as.data.frame(train.size.grid), 1, function(x)
  RunOneCNN(deep.NN,devices,train.all,test,n_train = x,error = TRUE,num_rounds = 20)$test.accuracy)

df <- data.frame(training_size = train.size.grid,
                 gold_standard = model1.accuracy,
                 silver_standard = model2.accuracy) %>%
  gather(., label_type, accuracy, -training_size)
head(df)

ggplot(df, aes(x = training_size, y = accuracy, group = label_type)) +
  geom_line(aes(col = label_type)) +
  ggtitle("Learning curves for different training labels")

# Add error to gold standard
p <- 30
mu0 <- runif(p, min = 0.1, max = 0.2)
mu1 <- runif(p, min = 0.3, max = 0.5)
Sigma0 <- diag(p)
Sigma1 <- diag(p)
train.y.error <- AddError(train.y,mu0,Sigma0,mu1,Sigma1)
train.y.error$error

# With predicted labels
lenet.model2 <- RunCNNModel(deep.NN,devices,
                            train.array,train.y.error$labels,
                            test.array,test.y)
lenet.model2$test.accuracy

# With predicted probabilities
pred.probs <- train.y.error$probs %>%
  cut(., 3)
lenet.model3 <- RunCNNModel(deep.NN,devices,
                            train.array,pred.probs,
                            test.array,test.y)
lenet.model3$test.accuracy

#graph.viz(lenet.model$model$symbol) # Visualizing architecture of CNN


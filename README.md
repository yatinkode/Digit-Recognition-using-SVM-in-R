# Digit-Recognition-using-SVM-in-R
You are required to develop a model using Support Vector Machine which should correctly classify the handwritten digits based on the pixel values given as features.

A classic problem in the field of pattern recognition is that of handwritten digit recognition. Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other digital devices. The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 


### Loading needed libraries

```R
load.libraries <- c('caret','kernlab','dplyr','readr','ggplot2','gridExtra','caTools','e1071')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dependences = TRUE)
sapply(load.libraries, require, character = TRUE)

#loading theme for ggplot
theme_gg <- function () { 
  theme_bw(base_size=10) %+replace% 
    theme(
      panel.background  = element_rect(fill="gray80", colour=NA),
      plot.background = element_rect(fill="gray96", colour=NA), 
      legend.background = element_rect(fill="white", colour=NA),
      legend.key = element_rect(fill="transparent", colour=NA),
      legend.position="bottom"
    )
}
```
### Data Loading and Understanding
```R
#loading dataset mnist_train.csv and mnist_test.csv
mnist_train<- read.csv("mnist_train.csv")
mnist_test<- read.csv("mnist_test.csv")

#combining train and test data into single dataset mnist_complete
mnist_complete<-rbind(mnist_train, setNames(rev(mnist_test), names(mnist_train)))

str(mnist_complete)
#69998 obs. of  785 variables

sum(is.na(mnist_complete))      #No NA values in the dataset

head(mnist_complete[1:10])
#X5 X0 X0.1 X0.2 X0.3 X0.4 X0.5 X0.6 X0.7 X0.8
#1  0  0    0    0    0    0    0    0    0    0
#2  4  0    0    0    0    0    0    0    0    0
#3  1  0    0    0    0    0    0    0    0    0
#4  9  0    0    0    0    0    0    0    0    0
#5  2  0    0    0    0    0    0    0    0    0
#6  1  0    0    0    0    0    0    0    0    0

tail(mnist_complete[1:10])
#      X5 X0 X0.1 X0.2 X0.3 X0.4 X0.5 X0.6 X0.7 X0.8
#69993  0  0    0    0    0    0    0    0    0    0
#69994  0  0    0    0    0    0    0    0    0    0
#69995  0  0    0    0    0    0    0    0    0    0
#69996  0  0    0    0    0    0    0    0    0    0
#69997  0  0    0    0    0    0    0    0    0    0
#69998  0  0    0    0    0    0    0    0    0    0
```

### Plotting Average Intensity of Digits
```R
#Calculating average pixel intensity
mnist_complete$pixavgint <- apply(mnist_complete[,-1], 1, mean) #takes the mean of each row in train

intbylabel <- aggregate (mnist_complete$pixavgint, by = list(mnist_complete$X5), FUN = mean)

#Plotting average intensity for each digit
ggplot(data=intbylabel, aes(x=Group.1, y = x)) +
  geom_bar(stat="identity") +
  labs(x = "Digit",y = "Average Intensity",title = "Fig : Average Intensity of Digits" )+theme_gg()
#The intensity for digit 1 is less compared to other digits. 
#Since digit 1 has less number of filled pixels compared to other digits
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/AverageIntensity.png)

### Plotting pixel Intensity of all digits
```R
plot0<-ggplot(data=subset(mnist_complete, X5 ==0),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg()+labs(x="intensity of 0") 

plot1<-ggplot(data=subset(mnist_complete, X5 ==1),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg()+labs(x="intensity of 1")  

plot2<-ggplot(data=subset(mnist_complete, X5 ==2),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg() +labs(x="intensity of 2")

plot3<-ggplot(data=subset(mnist_complete, X5 ==3),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg() +labs(x="intensity of 3")

plot4<-ggplot(data=subset(mnist_complete, X5 ==4),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg() +labs(x="intensity of 4")

plot5<-ggplot(data=subset(mnist_complete, X5 ==5),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg() +labs(x="intensity of 5")

plot6<-ggplot(data=subset(mnist_complete, X5 ==6),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg() +labs(x="intensity of 6")

plot7<-ggplot(data=subset(mnist_complete, X5 ==7),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg() +labs(x="intensity of 7")

plot8<-ggplot(data=subset(mnist_complete, X5 ==8),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg() +labs(x="intensity of 8")

plot9<-ggplot(data=subset(mnist_complete, X5 ==9),aes(pixavgint))+geom_histogram(binwidth = .75)+theme_gg() +labs(x="intensity of 9")

grid.arrange(plot0,plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9, ncol = 2,top="Fig : Intensity of different digits")
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/Intensityalldigits.png)

```R
#converting dependent variable to factor
mnist_complete$X5<-as.factor(mnist_complete$X5)

unique(mnist_complete$X5)
#Levels: 0 1 2 3 4 5 6 7 8 9
```

### Count of Digits in Dependent Variable X5
```R
#Checking occurences of X5 in the dataset
ggplot(mnist_complete, aes(x=as.factor(X5))) + geom_histogram(stat="count")+
  labs(x = "Digit",y = "Count of Digit",title = "Fig : Count of Digits" )+theme_gg()
#Most of the observations are for the digit 0, observations for all ther digits are around 5000
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/Countdigits.png)

### Looking at numbers formed from 0-9
```R
#function to flip around matrix fo digits
flip <- function(matrix){
  apply(matrix, 2, rev)
}
```
### Plotting different ways to write Digit 0
```R
digit0 <- mnist_complete[mnist_complete$X5 == 0, ]
digit0 <-  digit0[,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(digit0[i,])), nrow = 28))
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/0image.png)

### Plotting different ways to write Digit 1
```R
digit1 <- mnist_complete[mnist_complete$X5 == 1, ]
digit1 <-  digit1[ ,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(digit1[i,])), nrow = 28))
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/1image.png)

### Plotting different ways to write Digit 2
```R
digit2 <- mnist_complete[mnist_complete$X5 == 2, ]
digit2 <-  digit2[,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(digit2[i,])), nrow = 28))
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/2image.png)

### Plotting different ways to write Digit 3
```R
digit3 <- mnist_complete[mnist_complete$X5 == 3, ]
digit3 <-  digit3[,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit<- flip(matrix(rev(as.numeric(digit3[i,])), nrow = 28))
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/3image.png)

### Plotting different ways to write Digit 4
```R
digit4 <- mnist_complete[mnist_complete$X5 == 4, ]
digit4 <-  digit4[,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(digit4[i,])), nrow = 28))
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/4image.png)

### Plotting different ways to write Digit 5
```R
digit5 <- mnist_complete[mnist_complete$X5 == 5, ]
digit5 <-  digit5[,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(digit5[i,])), nrow = 28))
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/5image.png)

### Plotting different ways to write Digit 6
```R
digit6 <- mnist_complete[mnist_complete$X5 == 6, ]
digit6 <-  digit6[,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(digit6[i,])), nrow = 28)) #look at one digit
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/6image.png)

### Plotting different ways to write Digit 7
```R
digit7 <- mnist_complete[mnist_complete$X5 == 7, ]
digit7 <-  digit7[,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(digit7[i,])), nrow = 28))
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/7image.png)

### Plotting different ways to write Digit 8
```R
digit8 <- mnist_complete[mnist_complete$X5 == 8, ]
digit8 <-  digit8[,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(digit8[i,])), nrow = 28)) #look at one digit
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/8image.png)

### Plotting different ways to write Digit 9
```R
digit9 <- mnist_complete[mnist_complete$X5 == 9, ]
digit9 <-  digit9[ ,-c(1,786)]

par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(digit9[i,])), nrow = 28)) #look at one digit
  image(digit, col = grey.colors(255))
}
```
![data](https://github.com/yatinkode/Digit-Recognition-using-SVM-in-R/blob/main/images/9image.png)

### Preparing Data for Modelling
```R
#Due to availability of less computational resources we are considering only 15 % of data for further analysis
set.seed(100)
indices = sample.split(mnist_complete$X5, SplitRatio = 0.15)

mnist = mnist_complete[indices,]

#Remove Pixel Average Intensity column for model building
mnist<-mnist[,-786]

str(mnist)
#10500 obs. of  785 variables

sum(is.na(mnist))             #There are no NA values in the dataset
```

### Divide data into train and test
```R
indices = sample.split(mnist$X5, SplitRatio = 0.70)

train = mnist[indices,]
test = mnist[!(indices),]
```
### Model Building

```R
#------------------------------ Linear model - SVM  at Cost(C) = 1 -------------

#Converting dependent variables to factor
train$X5 <- as.factor(train$X5)
test$X5 <- as.factor(test$X5)

linear_model <- ksvm(X5~., data = train, scaled = F, kernel = "vanilladot")

print(linear_model)

# Predicting the model results 
evaluate_1<- predict(linear_model, test)

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
confusionMatrix(evaluate_1, test$X5)
#Accuracy : 0.9162
#Sensitivity : 0.903196
#Specificity : 0.990649

#              Class:0  Class:1  Class:2  Class:3  Class:4  Class:5  Class:6  Class:7  Class:8  Class:9
#Sensitivity:  0.9846   0.98350  0.88060  0.86232  0.90494  0.83607  0.95489  0.90780  0.84411  0.87313
#Specificity:  0.9881   0.99086  0.99028  0.99095  0.99064  0.98898  0.99376  0.99477  0.98960  0.98855

#--------------------------------------------- RBF MODEL - SVM at Cost(C) = 1 --------------------

rbf_model <- ksvm(X5~., data = train, scaled = F, kernel = "rbfdot")

print(rbf_model)                  #Get the values of C and sigma for further reference for cross validation
#C=1
#sigma= 1.59817433685909e-07 

# Predicting the model results 
evaluate_rbf<- predict(rbf_model, test)

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
confusionMatrix(evaluate_rbf, test$X5)

#Accuracy : 0.9571

#               Class:0  Class:1   Class:2  Class:3  Class:4  Class:5  Class:6  Class:7  Class:8   Class:9
#Sensitivity    0.9916   0.98680   0.92537  0.93478  0.93916  0.95492  0.95865  0.94681  0.95057   0.92164
#Specificity    0.9897   0.99543   0.99653  0.99722  0.99480  0.99518  0.99688  0.99791  0.99376   0.99410

#Sensitivity : 0.929341
#Specificity : 0.895633
```

### Model Evaluation

#### Hyperparameter tuning and Cross Validation  - Linear Model
```R
metric<-"Accuracy"
trainControl <- trainControl(method="cv", number=2)
trainControl <- trainControl(method="cv", number=5, verboseIter = TRUE)

# making a grid of C values. 
grid <- expand.grid(C=seq(1, 5, by=1))

# Performing 5-fold cross validation
fit.svm <- train(X5~., data=train, method="svmLinear", metric=metric, tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm)

# Plotting "fit.svm" results
plot(fit.svm,main = "Cost vs Accuracy for Linear Model")
# Accuracy remains same at all values of C
```



#### Hyperparameter tuning and Cross Validation  - RBF
```R
# We will use the train function from caret package to perform crossvalidation

# Making grid of "sigma" and C values.
# Range for Sigma and C values are being considered from the rbf model obtained earlier
grid <- expand.grid(.sigma=c(1.6e-07, 2.6e-07,3.6e-07,4.6e-07), .C=c(3,4,5))

# Performing 5-fold cross validation
fit.svm_radial <- train(X5~., data=train, method="svmRadial", metric=metric, tuneGrid=grid, trControl=trainControl)


# Printing cross validation result
print(fit.svm_radial)
# Best tune at sigma = 3.6e-07 & C=4

# Plotting fit.svm_radial results
plot(fit.svm_radial,main = "Sigma,Cost vs Accuracy for RBF Model")
#Accuracy is highest at C=3 and increases with increasing value of sigma
```

### Final Model
```R
f_model <- ksvm(X5~., data = train, scaled = F, kernel = "rbfdot",C=4,sigma=3.6e-07)

evaluate_f<- predict(f_model, test)

# Confusion Matrix - Finding accuracy, Sensitivity and specificity
confusionMatrix(evaluate_f, test$X5)
cf_final<-confusionMatrix(evaluate_f, test$X5)

mean(cf_final$byClass[,1])
#Sensitivity =  0.95967

mean(cf_final$byClass[,2])
#Specificity = 0.9960425
```

```R
############ Final Model ##########
#                                 #
#  Type               Radial      #
#  C                  4           #
#  Sigma              3.6e-07     #
#  Accuracy           0.9651      #
#  Sensitivity        0.95967     #
#  Specificity        0.9960425   #
#                                 #
###################################
```


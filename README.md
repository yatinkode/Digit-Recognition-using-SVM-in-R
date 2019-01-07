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


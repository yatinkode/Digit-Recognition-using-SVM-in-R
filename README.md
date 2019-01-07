# Digit-Recognition-using-SVM-in-R
You are required to develop a model using Support Vector Machine which should correctly classify the handwritten digits based on the pixel values given as features.

A classic problem in the field of pattern recognition is that of handwritten digit recognition. Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other digital devices. The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 

```#loading libraries
load.libraries <- c('caret','kernlab','dplyr','readr','ggplot2','gridExtra','caTools','e1071')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dependences = TRUE)
sapply(load.libraries, require, character = TRUE) ```

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


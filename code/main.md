---
title: "Group Lasso"
author: "MEMadsen"
date: "2018 May 2"
output: 
  html_document: 
    code_folding: hide
    fig_caption: yes
    keep_md: yes
    number_sections: yes
    theme: readable
    toc: yes
---


```r
getwd()
```

```
## [1] "/media/disc/Megasync/R/regularization/group_lasso/code"
```

```r
setwd("/media/disc/Megasync/R/regularization/group_lasso/code/")

# Load Packages
pkg.load <- c("autoEDA","bitops", "caret", "corrplot", "DataExplorer", "dplyr", "factoextra", "ggbiplot", "gglasso", "ggplot2", "glmnet", "h2o", "knitr", "markdown", "minerva", "mlbench","RColorBrewer", "RCurl", "reshape2", "rjson", "rmarkdown", "tools", "zoo", "rlang")



sapply(pkg.load, require, character.only = TRUE)
```

```
##      autoEDA       bitops        caret     corrplot DataExplorer 
##         TRUE         TRUE         TRUE         TRUE         TRUE 
##        dplyr   factoextra     ggbiplot      gglasso      ggplot2 
##         TRUE         TRUE         TRUE         TRUE         TRUE 
##       glmnet          h2o        knitr     markdown      minerva 
##         TRUE         TRUE         TRUE         TRUE         TRUE 
##      mlbench RColorBrewer        RCurl     reshape2        rjson 
##         TRUE         TRUE         TRUE         TRUE         TRUE 
##    rmarkdown        tools          zoo        rlang 
##         TRUE         TRUE         TRUE         TRUE
```



# Chapter 1


```r
# Document Updated last:
st.time.0.main <- Sys.time()
```




```r
## Optional -- Install Packages
#library(devtools)
#install_github("vqv/ggbiplot")

# Load Packages
pkg.load <- c("autoEDA","bitops", "caret", "corrplot", "DataExplorer", "dplyr", "factoextra", "ggbiplot", "gglasso", "ggplot2", "glmnet", "h2o", "knitr", "markdown", "minerva", "mlbench","RColorBrewer", "RCurl", "reshape2", "rjson", "rmarkdown", "tools", "zoo", "rlang")



sapply(pkg.load, require, character.only = TRUE)


cacheData = FALSE
cachePlot = TRUE
fig_width_11 = 11
fig_height_8 = 8
# fig.width=fig_width_11, fig.height=fig_height_8
fig_width_7 = 7 
fig_height_5 = 5 
# fig.width=fig_width_7, fig.height=fig_height_5

knitr::opts_chunk$set(progress = TRUE, fig.width=fig_width_11, fig.height=fig_height_8, message = FALSE, warning = FALSE, fig.align = "center", tidy=TRUE, tidy.opts=list(blank=TRUE, width.cutoff=65) )
# cache = TRUE, cache.lazy = TRUE, 
# opts_knit$set(self.contained=FALSE)
```


```r
ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/"
DATA.DIR <- paste(ROOT.DIR, "data", sep = "")
CHART.DIR <- paste(ROOT.DIR, "charts", sep = "")
CODE.DIR <- paste(ROOT.DIR, "code", sep = "")
```

# Sonar Description

This data set was used by Gorman and Sejnowski in their study of the classification of sonar signals using a neural network [Gorman and Sejnowski].  The task is to discriminate between sonar signals bounced off a metal cylinder and those bounced off a roughly cylindrical rock positioned on the ocean floor.


Each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number represents the energy within a particular frequency band, integrated over a certain period of time.  The integration aperture for higher frequencies occur later in time, since these frequencies are transmitted later during the chirp.


The label associated with each record contains the letter "R" if the object is a rock and "M" if it is a mine (metal cylinder). The numbers in the labels are in increasing order of aspect angle, but they do not encode the angle directly.

**Format**   
The data frame has 208 observations on 61 variables, all numerical and one nominal, e.g. Class.


```r
# Import Data data(Sonar) write.csv(Sonar, file = paste(DATA.DIR,
# 'sonar.csv', sep='') )
Sonar <- read.csv(file = paste(DATA.DIR, "sonar.csv", sep = "/"))
Sonar <- select(Sonar, -X)

# reomve rows with missing values Sonar_completeCase =
# Sonar[complete.cases(Sonar),] Sonar =
# Sonar[sample(nrow(Sonar)),]

# Assign Predictors and Response
y = Sonar$Class
X = Sonar[, -61]
```

# Exploratory Data Analysis {.tabset .tabset-fade}

## Initial Visualization

Without knowing anything about the data, my first 3 tasks are almost always:

* Are there missing values, and what is the missing data profile?   
* How does the categorical frequency for each discrete variable look like?  
* What is the distribution of each continuous variable?


The plot of missing data is not displayed because there is no missing data.  


```r
# plot_missing(Sonar)
```

### Categorical Frequency


The y-variable is well balanced across binary classes, e.g. rock or mine. Examining **Y** counts for the full data set is distributed as:

* Mines 111 
* Rocks 97


```r
plot_bar(Sonar)
```

<img src="main_files/figure-html/plt_bar-1.png" style="display: block; margin: auto;" />


### Continuous Distribution

Univariate histograms. The x-variables are not normally distributed..



```r
plot_histogram(Sonar)
```

### Box-plots 

Box-plots by Class label. Check for obviously different distribution of x-variables, by each class of Y.  


```r
plot_boxplot(Sonar, by = "Class")
```

<img src="main_files/figure-html/plt_boxplot-1.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/plt_boxplot-2.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/plt_boxplot-3.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/plt_boxplot-4.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/plt_boxplot-5.png" style="display: block; margin: auto;" />

```r
# plot_scatterplot(Sonar, by='Class')
```

## Data Overview


```r
dat_summary <- dataOverview(Sonar, outlierMethod = "tukey")


# Classification example:
power <- predictivePower(x = Sonar, y = "Class", outcomeType = "automatic")

overview <- autoEDA(x = Sonar, y = "Class")
```

```
## autoEDA | Setting color theme 
## autoEDA | Removing constant features 
## autoEDA | 0 constant features removed 
## autoEDA | Removing zero spread features 
## autoEDA | 0 zero spread features removed 
## autoEDA | Removing features containing majority missing values 
## autoEDA | 0 majority missing features removed 
## autoEDA | Cleaning data 
## autoEDA | Correcting sparse categorical feature levels 
## autoEDA | Sorting features 
## autoEDA | Binary classification outcome detected 
## autoEDA | Calculating feature predictive power 
## autoEDA | Visualizing data
```

<img src="main_files/figure-html/unnamed-chunk-5-1.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-2.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-3.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-4.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-5.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-6.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-7.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-8.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-9.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-10.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-11.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-12.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-13.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-14.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-15.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-16.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-17.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-18.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-19.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-20.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-21.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-22.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-23.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-24.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-25.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-26.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-27.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-28.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-29.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-30.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-31.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-32.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-33.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-34.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-35.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-36.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-37.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-38.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-39.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-40.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-41.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-42.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-43.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-44.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-45.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-46.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-47.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-48.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-49.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-50.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-51.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-52.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-53.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-54.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-55.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-56.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-57.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-58.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-59.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-60.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-61.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-5-62.png" style="display: block; margin: auto;" />

## Bivariate Frequency Polygon

There is significant overlap from the Rock/Mine signal on many of the predictor variables. It appears to be challenging to find strong predictors from a large portion of the variables. 


```r
ggplot(melt(Sonar[c(1:9, 61)], id.vars = "Class"), aes(x = value, 
    colour = Class)) + geom_freqpoly(aes(y = ..density..)) + scale_x_continuous(breaks = seq(0, 
    0.65, 0.1)) + facet_wrap(~variable)
```

<img src="main_files/figure-html/plt_bivar_freqpoly-1.png" style="display: block; margin: auto;" />

```r
ggplot(melt(Sonar[c(10:18, 61)], id.vars = "Class"), aes(x = value, 
    colour = Class)) + geom_freqpoly(aes(y = ..density..)) + scale_x_continuous(breaks = seq(0, 
    1, 0.25)) + facet_wrap(~variable)
```

<img src="main_files/figure-html/plt_bivar_freqpoly-2.png" style="display: block; margin: auto;" />

```r
ggplot(melt(Sonar[c(19:27, 61)], id.vars = "Class"), aes(x = value, 
    colour = Class)) + geom_freqpoly(aes(y = ..density..)) + scale_x_continuous(breaks = seq(0, 
    1, 0.25)) + facet_wrap(~variable)
```

<img src="main_files/figure-html/plt_bivar_freqpoly-3.png" style="display: block; margin: auto;" />

```r
ggplot(melt(Sonar[c(28:36, 61)], id.vars = "Class"), aes(x = value, 
    colour = Class)) + geom_freqpoly(aes(y = ..density..)) + scale_x_continuous(breaks = seq(0, 
    1, 0.25)) + facet_wrap(~variable)
```

<img src="main_files/figure-html/plt_bivar_freqpoly-4.png" style="display: block; margin: auto;" />

```r
ggplot(melt(Sonar[c(37:48, 61)], id.vars = "Class"), aes(x = value, 
    colour = Class)) + geom_freqpoly(aes(y = ..density..)) + scale_x_continuous(breaks = seq(0, 
    1, 0.25)) + facet_wrap(~variable)
```

<img src="main_files/figure-html/plt_bivar_freqpoly-5.png" style="display: block; margin: auto;" />

```r
ggplot(melt(Sonar[c(49:54, 61)], id.vars = "Class"), aes(x = value, 
    colour = Class)) + geom_freqpoly(aes(y = ..density..)) + scale_x_continuous(breaks = seq(0, 
    0.2, 0.05)) + facet_wrap(~variable)
```

<img src="main_files/figure-html/plt_bivar_freqpoly-6.png" style="display: block; margin: auto;" />

```r
ggplot(melt(Sonar[c(55:61)], id.vars = "Class"), aes(x = value, colour = Class)) + 
    geom_freqpoly(aes(y = ..density..)) + scale_x_continuous(breaks = seq(0, 
    0.05, 0.01)) + facet_wrap(~variable)
```

<img src="main_files/figure-html/plt_bivar_freqpoly-7.png" style="display: block; margin: auto;" />


## Maximal Information Coefficient

The "Maximal Information Coefficient" (MIC) is able to describe the correlation between paired variables regardless of linear or nonlinear relationship. 



```r
micValues <- mine(x = Sonar[, -61], y = ifelse(Sonar$Class == "M", 
    1, 0), alpha = 0.7)

# Analysis
M <- micValues$MIC
P <- cor(x = Sonar[, -61], y = ifelse(Sonar$Class == "M", 1, 0))

res <- data.frame(MIC = c(micValues$MIC))
# res <- data.frame(M)
rownames(res) <- rownames(micValues$MIC)
res$MIC_Rank <- nrow(res) - rank(res$MIC, ties.method = "first") + 
    1
res$Pearson <- P
res$Pearson_Rank <- nrow(res) - rank(abs(res$Pearson), ties.method = "first") + 
    1
res <- res[order(res$MIC_Rank), ]
head(res, n = 10)
```

```
##           MIC MIC_Rank    Pearson Pearson_Rank
## V12 0.5104860        1  0.3922455            2
## V11 0.5068413        2  0.4328549            1
## V49 0.4147951        3  0.3513123            3
## V9  0.4082225        4  0.3214484            7
## V10 0.4054348        5  0.3411418            4
## V48 0.4019841        6  0.3293332            6
## V5  0.3923728        7  0.2222318           22
## V47 0.3888595        8  0.3016967           10
## V37 0.3866557        9 -0.2090547           23
## V46 0.3699272       10  0.3056100            9
```

```r
tail(res, n = 10)
```

```
##           MIC MIC_Rank       Pearson Pearson_Rank
## V55 0.2856680       51  0.0956385371           42
## V27 0.2847828       52  0.0549968239           49
## V6  0.2840402       53  0.1323265038           35
## V59 0.2743641       54  0.1308259367           36
## V41 0.2712223       55  0.0209421165           55
## V34 0.2687419       56 -0.1720102902           30
## V54 0.2686506       57  0.1826874301           28
## V53 0.2602140       58  0.1418711269           33
## V57 0.2470229       59  0.0009328276           60
## V56 0.2286102       60  0.1293405549           37
```


```r
# Plot png('MIC_vs_Pearson.png', width=7.5, height=3.5, res=400,
# units='in', type='cairo')
op <- par(mfrow = c(1, 2), mar = c(4, 4, 1, 1))
plot(MIC ~ abs(Pearson), res, pch = 21, col = 4, bg = 5)
plot(MIC_Rank ~ Pearson_Rank, res, pch = 21, col = 4, bg = 5)
```

<img src="main_files/figure-html/plt_MIC-1.png" style="display: block; margin: auto;" />

```r
par(op)
# dev.off()
```

### Correlation 


```r
plot_correlation(Sonar)
```

<img src="main_files/figure-html/correlation-1.png" style="display: block; margin: auto;" />


### Hierarchial Clustering


```r
# Dissimilarity matrix
d = dist(t(X), method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 = hclust(d, method = "complete")

# Apply Elbow rule
fviz_nbclust(X, FUN = hcut, method = "wss")  # It seems 4 - 6 groups may be appropriate
```

<img src="main_files/figure-html/cluster_hier-1.png" style="display: block; margin: auto;" />

```r
# Plot dendrogram with 5 selected groups
plot(hc1, cex = 0.6, hang = -1)
rect.hclust(hc1, k = 5, border = 2:5)
```

<img src="main_files/figure-html/cluster_hier-2.png" style="display: block; margin: auto;" />

# Caret Models



```r
# source('/media/disc/Megasync/R/regularization/group_lasso/code/grouped_lasso.Rmd')
```


# Machine Learning  {.tabset .tabset-fade}

The machine learning package H2O is leveraged using the R interface. 




This child file `deep_learn_h20_roxy.R` uses roxygen formatting on a raw R file.


```r
# spin_child('/media/disc/Megasync/R/regularization/group_lasso/code/spinner_test.R')
{
    {
        knitr::spin_child("/media/disc/Megasync/R/regularization/group_lasso/code/deep_learn_h20_roxy.R")
    }
}
```


 LIBRARIES for running h2o  

```r
# Download packages that H2O depends on.
pkgs <- c("pROC", "RCurl", "jsonlite")
for (pkg in pkgs) {
    if (!(pkg %in% rownames(installed.packages()))) {
        install.packages(pkg)
    }
}
# Load libs
pkg.load <- c("AppliedPredictiveModeling", "bitops", "caret", "dplyr", 
    "ggbiplot", "h2o", "mlbench", "pROC", "RCurl", "rjson", "ROCR", 
    "statmod", "tools")
sapply(pkg.load, require, character.only = TRUE)
```

**Helper File** for DeepLearning Model

```r
source("/media/disc/Megasync/R/regularization/group_lasso/code/helper_h2o.R")
```

```r
ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/"
DATA.DIR <- paste(ROOT.DIR, "data", sep = "")
CODE.DIR <- paste(ROOT.DIR, "code", sep = "")
```

**MAKE RESULTS REPEATABLE**

```r
seed = 77
set.seed(seed)
```

## INITIALIZE H2O 

H2O is a Java application. It runs a web server and interacts with R
via REST-like web services. The command below starts the H2O. The nthreads
parameter value -1 tells H2O to run with as many threads as there are
CPU cores.

First connect to a local H2O instance from RStudio using all CPUs and 14 gigabytes of memory. 
 

```r
h2o <- h2o.init(nthreads = -1, max_mem_size = "14g")
```

```
##  Connection successful!
## 
## R is connected to the H2O cluster: 
##     H2O cluster uptime:         1 hours 39 minutes 
##     H2O cluster timezone:       America/New_York 
##     H2O data parsing timezone:  UTC 
##     H2O cluster version:        3.18.0.8 
##     H2O cluster version age:    19 days  
##     H2O cluster name:           H2O_started_from_R_npcomplete_xgj592 
##     H2O cluster total nodes:    1 
##     H2O cluster total memory:   12.26 GB 
##     H2O cluster total cores:    4 
##     H2O cluster allowed cores:  4 
##     H2O cluster healthy:        TRUE 
##     H2O Connection ip:          localhost 
##     H2O Connection port:        54321 
##     H2O Connection proxy:       NA 
##     H2O Internal Security:      FALSE 
##     H2O API Extensions:         XGBoost, Algos, AutoML, Core V3, Core V4 
##     R Version:                  R version 3.4.4 (2018-03-15)
```

LOAD SONAR DATA AND CREATE TRAINING AND TEST SETS  

```r
# data(Sonar) write.csv(Sonar, file = paste(DATA.DIR,
# 'sonar_dl.csv', sep='/'), row.names=FALSE )
Sonar <- read.csv(file = paste(DATA.DIR, "sonar_dl.csv", sep = "/"))
# 
pathToData <- paste0(normalizePath(DATA.DIR), "/sonar_dl.csv")
```

1. Load data into an H2O data frame
1. Optional: Convert the H2O data frame into an R data frame for later use.
1. Create the feature vectors by dropping the classifications.

```r
# sonar.hex =
# h2o.importFile('/media/disc/Megasync/R/regularization/group_lasso/data/sonar_dl.csv')
sonar.hex <- h2o.importFile(path = pathToData, destination_frame = "sonar.hex")
# 
sonar.df = as.data.frame(sonar.hex)
# Create the feature vectors by dropping the classifications
classVariableIndex = 61
sonar.features = sonar.df[, -classVariableIndex]
# Rename the classification varible to something meaningful
names(sonar.df)[classVariableIndex] = "Class"
```

Use H2O to create the traning and test data sets from the original data.

* 80% Train 

* 20% Test

Pass seed value generated previously to make the results repeatable.

```r
sonar.split = h2o.splitFrame(data = sonar.hex, ratios = 0.8, seed = seed)
# Create named varaibles for the traning and test data
parts <- sonar.split[[1]]
parts <- h2o.splitFrame(parts, 1/6)
sonar.valid <- parts[[1]]
sonar.train <- parts[[2]]
sonar.test = sonar.split[[2]]
# Create R data frames from the H2O data frames
sonar.train.df = as.data.frame(sonar.train)
names(sonar.train.df)[classVariableIndex] = "Class"
sonar.test.df = as.data.frame(sonar.test)
names(sonar.test.df)[classVariableIndex] = "Class"
```

### TRELLIS PLOT   

Check data quality to ensure everything went good so far. 
Create a trellis plot using all 60 features. Notice that it is a useless mess.

```r
transparentTheme(trans = 0.5)
featurePlot <- featurePlot(x = sonar.features, y = sonar.df$Class, 
    plot = "pairs", pscales = FALSE, cex = 0.05, auto.key = list(columns = 2))
featurePlot
```

<img src="main_files/figure-html/plt_trellis-1.png" style="display: block; margin: auto;" />

## PRINCIPAL COMPONENT ANALYSIS  

Generally, anything with more than 10 features is considered high dimensional data.
Text mining and genomic microarrays have thousands of dimensions and they are the
go-to examples of high dimensional data.
---

```r
# Use principal components to reduce the dimensionality of the
# features.
sonar.princomp = prcomp(sonar.features, center = TRUE, scale = TRUE, 
    retx = TRUE)
sonar.princomp.7 = prcomp(sonar.features, center = TRUE, scale = TRUE, 
    tol = 0.4, retx = TRUE)
```

Plot the first three principal components. Together, the first three
principle components accounts for almost 50% of the variance in the data!
That means they are very rich in information. After the PCA transformation,
The original data set contains 60 features. If we think of the first three principal
components as features, then 3 features can account for 50% of the information in the
original data set. That means the the first three principle components consume only
3/60, or just 5% of the space as the original data, but contains almost half the signal.
The upshot is that PCA is a kind of lossy data compression! 

### PCA Cumulative Variance

Look at cumulative proportion of the variance explained by the principle components:

```r
summary(sonar.princomp.7)
```

```
## Importance of first k=7 (out of 60) components:
##                           PC1    PC2    PC3     PC4     PC5     PC6
## Standard deviation     3.4940 3.3672 2.2649 1.84595 1.73328 1.56173
## Proportion of Variance 0.2035 0.1890 0.0855 0.05679 0.05007 0.04065
## Cumulative Proportion  0.2035 0.3924 0.4779 0.53473 0.58480 0.62545
##                            PC7
## Standard deviation     1.40264
## Proportion of Variance 0.03279
## Cumulative Proportion  0.65824
```

```r
# PCA TRELLIS PLOT
transparentTheme(trans = 0.3)
numberOfPrincipalComponentsToPlot = 3
```

Plotting the principle components shows us that the rocks and mines have a lot of
overlap. The more overlaps, the harder it will be for the machine learning algorithm
to distinguish the difference between rocks and mines. However, it is easy to see the
two classes do not overlap entirely; there is definitely some signal for the algorithm
to lock onto.

```r
featurePlot(x = sonar.princomp$x[, 1:numberOfPrincipalComponentsToPlot], 
    y = sonar.df$Class, plot = "pairs", pscales = FALSE, auto.key = list(columns = 2))
```

<img src="main_files/figure-html/plt_PCA_feature-1.png" style="display: block; margin: auto;" />

### PLOT FIRST AND SECOND PRINCIPLE COMPONENTS 

Use a special libary to plot the first principle component against the second
principle component. Draw a ellipses around the observations. 

* Observations inside the ellipse are less than 1 std dev away from mean.

```r
g <- ggbiplot(
  sonar.princomp,
  groups = sonar.df$Class,
  var.axes = FALSE,
  #
  ellipse = TRUE
  )
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal',
               legend.position = 'top')
print(g)
```

<img src="main_files/figure-html/plt_PCA_loadings-1.png" style="display: block; margin: auto;" />

# Trees {.tabset .tabset-fade}

## Decision Tree

Let's see how a decision tree does. Having a baseline to compare accuracy is useful. 

```r
run.DTREE = 0
if (run.DTREE == 1) {
    time.dtree.fit <- Sys.time()
    fit.d.tree = train(Class ~ ., sonar.train.df, method = "ctree")
    time.dtree.end <- Sys.time()
    time.dtree.diff <- time.dtree.end - time.dtree.fit
    
    saveRDS(time.dtree.diff, file = paste(DATA.DIR, "time.dtree.diff.rds", 
        sep = "/"))
    save(fit.d.tree, file = paste(DATA.DIR, "fit.d.tree.Rdata", sep = "/"))
    saveRDS(fit.d.tree, file = paste(DATA.DIR, "fit.d.tree.rds", sep = "/"))
}
# 
time.dtree.diff <- readRDS(file = paste(DATA.DIR, "time.dtree.diff.rds", 
    sep = "/"))
# fit.cv.ridge <- readRDS(file = paste(DATA.DIR,
# 'fit.cv.ridge.rds', sep='/'))
load(file = paste(DATA.DIR, "fit.d.tree.Rdata", sep = "/"))
```

Visually inspect the tree.

```r
plot(fit.d.tree$finalModel)
```

<img src="main_files/figure-html/unnamed-chunk-6-1.png" style="display: block; margin: auto;" />

Evaluate the decision tree performance

```r
pred.d.tree = predict(fit.d.tree, sonar.test.df[, -classVariableIndex])
res.d.tree <- caret::confusionMatrix(pred.d.tree, sonar.test.df$Class)
```

## AUC results

The decision tree area-under-curve (AUC) results look good. Reported 84%  

```r
pred.d.tree.probabilities = predict(fit.d.tree, sonar.test.df[, -classVariableIndex], 
    type = "prob")
d.tree.roc = roc(sonar.test.df$Class, pred.d.tree.probabilities$M)
# Area under the curve: 0.8433
plot(d.tree.roc)
```

<img src="main_files/figure-html/auc_results-1.png" style="display: block; margin: auto;" />

## Random Forest and GBM

  
 1. Gradient Boosted Model AUC=91%
 1. Random Forest
 
Ensemble learners (EL) are usually composed of decision trees.
 An EL combines multiple weak learners into a single strong learner.
 See bias-variance tradeoff for better understanding.  
 

```r
model.gbm.exp <- h2o.experiment(h2o.gbm)
```

```
##   |                                                                         |                                                                 |   0%  |                                                                         |=====                                                            |   8%  |                                                                         |=================================================================| 100%
##   |                                                                         |                                                                 |   0%  |                                                                         |=================================================================| 100%
```

<img src="main_files/figure-html/gbm_fit-1.png" style="display: block; margin: auto;" />

```
## H2OBinomialMetrics: gbm
## 
## MSE:  0.1780894
## RMSE:  0.4220065
## LogLoss:  0.5448465
## Mean Per-Class Error:  0.2083333
## AUC:  0.8529412
## Gini:  0.7058824
## 
## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
##         M  R    Error    Rate
## M      14 10 0.416667  =10/24
## R       0 17 0.000000   =0/17
## Totals 14 27 0.243902  =10/41
## 
## Maximum Metrics: Maximum metrics at their respective thresholds
##                         metric threshold    value idx
## 1                       max f1  0.261945 0.772727  26
## 2                       max f2  0.261945 0.894737  26
## 3                 max f0point5  0.817075 0.779221  14
## 4                 max accuracy  0.817075 0.804878  14
## 5                max precision  0.992510 1.000000   0
## 6                   max recall  0.261945 1.000000  26
## 7              max specificity  0.992510 1.000000   0
## 8             max absolute_mcc  0.261945 0.606040  26
## 9   max min_per_class_accuracy  0.690129 0.708333  19
## 10 max mean_per_class_accuracy  0.261945 0.791667  26
## 
## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```

```r
m.gbm0 <- h2o.gbm(1:60, 61, sonar.train, nfolds = 0, model_id = "GBM_default")
```

```
##   |                                                                         |                                                                 |   0%  |                                                                         |======                                                           |  10%  |                                                                         |=================================================================| 100%
```

```r
m.gbm0
```

```
## Model Details:
## ==============
## 
## H2OBinomialModel: gbm
## Model ID:  GBM_default 
## Model Summary: 
##   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
## 1              50                       50                9713         4
##   max_depth mean_depth min_leaves max_leaves mean_leaves
## 1         5    4.88000          8         12    10.52000
## 
## 
## H2OBinomialMetrics: gbm
## ** Reported on training data. **
## 
## MSE:  0.000729008
## RMSE:  0.02700015
## LogLoss:  0.02279554
## Mean Per-Class Error:  0
## AUC:  1
## Gini:  1
## 
## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
##         M  R    Error    Rate
## M      70  0 0.000000   =0/70
## R       0 66 0.000000   =0/66
## Totals 70 66 0.000000  =0/136
## 
## Maximum Metrics: Maximum metrics at their respective thresholds
##                         metric threshold    value idx
## 1                       max f1  0.923763 1.000000  65
## 2                       max f2  0.923763 1.000000  65
## 3                 max f0point5  0.923763 1.000000  65
## 4                 max accuracy  0.923763 1.000000  65
## 5                max precision  0.992702 1.000000   0
## 6                   max recall  0.923763 1.000000  65
## 7              max specificity  0.992702 1.000000   0
## 8             max absolute_mcc  0.923763 1.000000  65
## 9   max min_per_class_accuracy  0.923763 1.000000  65
## 10 max mean_per_class_accuracy  0.923763 1.000000  65
## 
## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```

```r
# Random forest. A good ensemble learner.
model.rf.exp <- h2o.experiment(h2o.randomForest)
```

```
##   |                                                                         |                                                                 |   0%  |                                                                         |=========                                                        |  14%  |                                                                         |=================================================================| 100%
##   |                                                                         |                                                                 |   0%  |                                                                         |=================================================================| 100%
```

<img src="main_files/figure-html/randForest_fit-1.png" style="display: block; margin: auto;" />

```
## H2OBinomialMetrics: drf
## 
## MSE:  0.1634732
## RMSE:  0.4043182
## LogLoss:  0.4878345
## Mean Per-Class Error:  0.2083333
## AUC:  0.8480392
## Gini:  0.6960784
## 
## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
##         M  R    Error    Rate
## M      14 10 0.416667  =10/24
## R       0 17 0.000000   =0/17
## Totals 14 27 0.243902  =10/41
## 
## Maximum Metrics: Maximum metrics at their respective thresholds
##                         metric threshold    value idx
## 1                       max f1  0.340000 0.772727  18
## 2                       max f2  0.340000 0.894737  18
## 3                 max f0point5  0.820000 0.731707   4
## 4                 max accuracy  0.500000 0.780488  14
## 5                max precision  0.980000 1.000000   0
## 6                   max recall  0.340000 1.000000  18
## 7              max specificity  0.980000 1.000000   0
## 8             max absolute_mcc  0.340000 0.606040  18
## 9   max min_per_class_accuracy  0.500000 0.708333  14
## 10 max mean_per_class_accuracy  0.500000 0.795343  14
## 
## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```

```r
m.rf0 <- h2o.randomForest(1:60, 61, sonar.train, nfolds = 0, model_id = "RF_default")
```

```
##   |                                                                         |                                                                 |   0%  |                                                                         |==========                                                       |  16%  |                                                                         |=================================================================| 100%
```

```r
summary(m.rf0)
```

```
## Model Details:
## ==============
## 
## H2OBinomialModel: drf
## Model Key:  RF_default 
## Model Summary: 
##   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
## 1              50                       50               12848         5
##   max_depth mean_depth min_leaves max_leaves mean_leaves
## 1        10    6.86000         11         21    15.52000
## 
## H2OBinomialMetrics: drf
## ** Reported on training data. **
## ** Metrics reported on Out-Of-Bag training samples **
## 
## MSE:  0.1231583
## RMSE:  0.3509392
## LogLoss:  0.3886211
## Mean Per-Class Error:  0.1547619
## AUC:  0.9245671
## Gini:  0.8491342
## 
## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
##         M  R    Error     Rate
## M      60 10 0.142857   =10/70
## R      11 55 0.166667   =11/66
## Totals 71 65 0.154412  =21/136
## 
## Maximum Metrics: Maximum metrics at their respective thresholds
##                         metric threshold    value idx
## 1                       max f1  0.500000 0.839695  40
## 2                       max f2  0.294118 0.905292  59
## 3                 max f0point5  0.600000 0.866142  31
## 4                 max accuracy  0.500000 0.845588  40
## 5                max precision  1.000000 1.000000   0
## 6                   max recall  0.200000 1.000000  67
## 7              max specificity  1.000000 1.000000   0
## 8             max absolute_mcc  0.500000 0.690850  40
## 9   max min_per_class_accuracy  0.500000 0.833333  40
## 10 max mean_per_class_accuracy  0.500000 0.845238  40
## 
## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
## 
## 
## 
## Scoring History: 
##             timestamp   duration number_of_trees training_rmse
## 1 2018-05-09 12:22:58  0.001 sec               0              
## 2 2018-05-09 12:22:58  0.008 sec               1       0.50000
## 3 2018-05-09 12:22:58  0.012 sec               2       0.51841
## 4 2018-05-09 12:22:58  0.016 sec               3       0.46927
## 5 2018-05-09 12:22:58  0.022 sec               4       0.46876
##   training_logloss training_auc training_lift
## 1                                            
## 2          8.63469      0.75681       1.34387
## 3          8.68668      0.73232       1.40989
## 4          6.83746      0.78077       1.51111
## 5          6.14986      0.78305       1.56785
##   training_classification_error
## 1                              
## 2                       0.25000
## 3                       0.27500
## 4                       0.23711
## 5                       0.22936
## 
## ---
##              timestamp   duration number_of_trees training_rmse
## 46 2018-05-09 12:22:58  0.183 sec              45       0.35578
## 47 2018-05-09 12:22:58  0.187 sec              46       0.35508
## 48 2018-05-09 12:22:58  0.191 sec              47       0.35510
## 49 2018-05-09 12:22:58  0.195 sec              48       0.35358
## 50 2018-05-09 12:22:58  0.199 sec              49       0.35321
## 51 2018-05-09 12:22:58  0.203 sec              50       0.35094
##    training_logloss training_auc training_lift
## 46          0.39698      0.91526       2.06061
## 47          0.39439      0.91623       2.06061
## 48          0.39450      0.91558       2.06061
## 49          0.39208      0.91861       2.06061
## 50          0.39156      0.91937       2.06061
## 51          0.38862      0.92457       2.06061
##    training_classification_error
## 46                       0.17647
## 47                       0.16912
## 48                       0.17647
## 49                       0.19118
## 50                       0.17647
## 51                       0.15441
## 
## Variable Importances: (Extract with `h2o.varimp`) 
## =================================================
## 
## Variable Importances: 
##   variable relative_importance scaled_importance percentage
## 1      V11          111.966743          1.000000   0.093786
## 2      V10           78.738930          0.703235   0.065954
## 3       V9           71.330414          0.637068   0.059748
## 4      V12           64.250557          0.573836   0.053818
## 5      V52           53.300331          0.476037   0.044646
## 
## ---
##    variable relative_importance scaled_importance percentage
## 55      V18            5.619047          0.050185   0.004707
## 56      V34            5.367100          0.047935   0.004496
## 57      V25            5.046689          0.045073   0.004227
## 58       V4            4.863054          0.043433   0.004073
## 59      V56            3.148810          0.028123   0.002638
## 60      V33            1.383275          0.012354   0.001159
```

```r
h2o.performance(m.rf0, sonar.test)
```

```
## H2OBinomialMetrics: drf
## 
## MSE:  0.1668098
## RMSE:  0.4084235
## LogLoss:  0.4994168
## Mean Per-Class Error:  0.1838235
## AUC:  0.8529412
## Gini:  0.7058824
## 
## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
##         M  R    Error   Rate
## M      18  6 0.250000  =6/24
## R       2 15 0.117647  =2/17
## Totals 20 21 0.195122  =8/41
## 
## Maximum Metrics: Maximum metrics at their respective thresholds
##                         metric threshold    value idx
## 1                       max f1  0.520000 0.789474  14
## 2                       max f2  0.440000 0.860215  16
## 3                 max f0point5  0.520000 0.742574  14
## 4                 max accuracy  0.520000 0.804878  14
## 5                max precision  0.960000 1.000000   0
## 6                   max recall  0.360000 1.000000  20
## 7              max specificity  0.960000 1.000000   0
## 8             max absolute_mcc  0.520000 0.623254  14
## 9   max min_per_class_accuracy  0.580000 0.750000  13
## 10 max mean_per_class_accuracy  0.520000 0.816176  14
## 
## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```

```r
res <- compareModels(c(m.gbm0, m.rf0), sonar.test)
round(res[, "AUC", ], 3)
# 
compareModels <- function(models, test, labels = NULL) {
    # Use model IDs as default labels, if not given
    if (is.null(labels)) {
        labels <- lapply(models, function(m) m@model_id)
    }
    
    res <- sapply(models, function(m) {
        mcmsT <- m@model$training_metrics@metrics$max_criteria_and_metric_scores
        mcmsV <- m@model$validation_metrics@metrics$max_criteria_and_metric_scores
        maix <- which(mcmsT$metric == "max accuracy")  #4 (at the time of writing)
        th <- mean(mcmsT[maix, "threshold"], mcmsV[maix, "threshold"])
        
        pf <- h2o.performance(m, test)
        tms <- pf@metrics$thresholds_and_metric_scores
        ix <- apply(outer(th, tms$threshold, "<="), 1, sum)
        if (ix < 1) 
            ix <- 1  #Use first entry if less than all of them
        
        matrix(c(h2o.auc(m, TRUE, TRUE), pf@metrics$AUC, mcmsT[maix, 
            "value"], mcmsV[maix, "value"], tms[ix, "accuracy"], h2o.logloss(m, 
            TRUE, TRUE), pf@metrics$logloss, h2o.mse(m, TRUE, TRUE), 
            pf@metrics$MSE), ncol = 4)
    }, simplify = "array")
    
    dimnames(res) <- list(c("train", "valid", "test"), c("AUC", "Accuracy", 
        "logloss", "MSE"), labels)
    
    res
}
```

## GLM

 Generalized linear model.
  
1. GLM AUC=94% 
 

```r
h2o.experiment(h2o.glm, list(family = "binomial"))
```

```
##   |                                                                         |                                                                 |   0%  |                                                                         |=================================================================| 100%
##   |                                                                         |                                                                 |   0%  |                                                                         |=================================================================| 100%
```

<img src="main_files/figure-html/glm_fit-1.png" style="display: block; margin: auto;" />

```
## H2OBinomialMetrics: glm
## 
## MSE:  0.1854945
## RMSE:  0.4306907
## LogLoss:  0.5477175
## Mean Per-Class Error:  0.2009804
## AUC:  0.8235294
## Gini:  0.6470588
## R^2:  0.2357444
## Residual Deviance:  44.91284
## AIC:  106.9128
## 
## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
##         M  R    Error   Rate
## M      20  4 0.166667  =4/24
## R       4 13 0.235294  =4/17
## Totals 24 17 0.195122  =8/41
## 
## Maximum Metrics: Maximum metrics at their respective thresholds
##                         metric threshold    value idx
## 1                       max f1  0.624500 0.764706  16
## 2                       max f2  0.314302 0.824742  28
## 3                 max f0point5  0.737546 0.789474   9
## 4                 max accuracy  0.692087 0.804878  14
## 5                max precision  0.899520 1.000000   0
## 6                   max recall  0.061872 1.000000  36
## 7              max specificity  0.899520 1.000000   0
## 8             max absolute_mcc  0.624500 0.598039  16
## 9   max min_per_class_accuracy  0.624500 0.764706  16
## 10 max mean_per_class_accuracy  0.624500 0.799020  16
## 
## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```

```
## H2OBinomialMetrics: glm
## 
## MSE:  0.1854945
## RMSE:  0.4306907
## LogLoss:  0.5477175
## Mean Per-Class Error:  0.2009804
## AUC:  0.8235294
## Gini:  0.6470588
## R^2:  0.2357444
## Residual Deviance:  44.91284
## AIC:  106.9128
## 
## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
##         M  R    Error   Rate
## M      20  4 0.166667  =4/24
## R       4 13 0.235294  =4/17
## Totals 24 17 0.195122  =8/41
## 
## Maximum Metrics: Maximum metrics at their respective thresholds
##                         metric threshold    value idx
## 1                       max f1  0.624500 0.764706  16
## 2                       max f2  0.314302 0.824742  28
## 3                 max f0point5  0.737546 0.789474   9
## 4                 max accuracy  0.692087 0.804878  14
## 5                max precision  0.899520 1.000000   0
## 6                   max recall  0.061872 1.000000  36
## 7              max specificity  0.899520 1.000000   0
## 8             max absolute_mcc  0.624500 0.598039  16
## 9   max min_per_class_accuracy  0.624500 0.764706  16
## 10 max mean_per_class_accuracy  0.624500 0.799020  16
## 
## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```

```r
fTable1 <- kable(forec.table.h6.p1, digits = 0, format = "markdown", 
    padding = 2, caption = "Forecast results for candidate models")
fTable2 <- kable(forec.table.h6.p2, digits = 0, format = "markdown", 
    padding = 2, caption = "Forecast results for candidate models")
print(fTable1)
print(fTable2)
```

## Deep Learning Model Fit

Deep learning (DL) can be considered as a neural network that includes extra hidden layers. These hidden layers account for nonlinear relationships and latent variance. Raw input is automatically converted to good features. The increased accuracy
comes at a computationally expensive cost. Tuning the model parameters will have 
dramatic effects on the accuracy. An untuned DL model will generally perform poorly, with low accuracy on test set predictions.

```r
dl.fit <- h2o.experiment(h2o.deeplearning)
```

```
##   |                                                                         |                                                                 |   0%  |                                                                         |======                                                           |  10%  |                                                                         |=================================================================| 100%
##   |                                                                         |                                                                 |   0%  |                                                                         |=================================================================| 100%
```

<img src="main_files/figure-html/deep_learning_fit-1.png" style="display: block; margin: auto;" />

```
## H2OBinomialMetrics: deeplearning
## 
## MSE:  0.2018155
## RMSE:  0.4492388
## LogLoss:  1.225112
## Mean Per-Class Error:  0.2303922
## AUC:  0.8014706
## Gini:  0.6029412
## 
## Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
##         M  R    Error   Rate
## M      20  4 0.166667  =4/24
## R       5 12 0.294118  =5/17
## Totals 25 16 0.219512  =9/41
## 
## Maximum Metrics: Maximum metrics at their respective thresholds
##                         metric threshold    value idx
## 1                       max f1  0.395169 0.727273  15
## 2                       max f2  0.000024 0.850000  31
## 3                 max f0point5  0.875571 0.754717   8
## 4                 max accuracy  0.395169 0.780488  15
## 5                max precision  0.999981 1.000000   0
## 6                   max recall  0.000024 1.000000  31
## 7              max specificity  0.999981 1.000000   0
## 8             max absolute_mcc  0.395169 0.544581  15
## 9   max min_per_class_accuracy  0.395169 0.705882  15
## 10 max mean_per_class_accuracy  0.395169 0.769608  15
## 
## Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
```



# Model Results  (MAIN doc)

Comparing performance accuracy from models. 
 


Model           |Accuracy                               | Run Time            |No. of Predictors|
----------------|---------------------------------------|---------------------|-----------------|
Decision Tree   | 0.8  | 4.64 | 6   |



# References 


* Gorman,  R.  P.,  and  Sejnowski,  T.  J.  (1988).   "Analysis  of  Hidden  Units  in  a  Layered  Network Trained to Classify Sonar Targets" in Neural Networks, Vol. 1, pp. 75-89.


* [H2O Deep Learning][https://github.com/ahoffer/badger-beats] Liberal use of this GitHub page is utilized for the DeepLearning/H2O/Machine Learning section of this document.

* [Maximal Information Coefficient][http://minepy.readthedocs.io/en/latest/details.html] http://www.exploredata.net/

**Brief Review ROC plots.** 

Explain ROC in terms of classifying rocks vs mines. The more tolerant you are of "slop" (false positives), the more you can be sure you aren't missing any true positives. The more willing you are to accept the computer misclassifying, the less likely you are to miss-classify a mine.

PS: ROC curve technical background requires some understanding. I've been meaning to read these articles about it:

* http://blogs.sas.com/content/iml/2011/07/29/computing-an-roc-curve-from-basic-principles.html

* http://www.dataschool.io/roc-curves-and-auc-explained/

## Run Time 

1. Start time
1. End time


```r
st.time.0.main
```

```
## [1] "2018-05-09 12:21:49 EDT"
```

```r
end.time.0.main <- Sys.time()
end.time.0.main
```

```
## [1] "2018-05-09 12:23:01 EDT"
```

```r
end.time.0.main - st.time.0.main
```

```
## Time difference of 1.202289 mins
```

**Document Settings **



```r
getwd()
```

```
## [1] "/media/disc/Megasync/R/regularization/group_lasso/code"
```

```r
list.files(getwd())
```

```
##  [1] "deep_learn_h20_roxy.R"      "deep_learn_h2o_plain.R"    
##  [3] "grouped_lasso.Rmd"          "grouped_lasso_YMLonTop.Rmd"
##  [5] "h20.Rmd"                    "h20_YMLonTop.Rmd"          
##  [7] "helper_h2o.R"               "main_cache"                
##  [9] "main_files"                 "main.Rmd"
```

```r
sessionInfo()
```

```
## R version 3.4.4 (2018-03-15)
## Platform: x86_64-pc-linux-gnu (64-bit)
## Running under: Ubuntu 16.04.4 LTS
## 
## Matrix products: default
## BLAS: /usr/lib/libblas/libblas.so.3.6.0
## LAPACK: /usr/lib/lapack/liblapack.so.3.6.0
## 
## locale:
##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       
## 
## attached base packages:
##  [1] stats4    tools     grid      stats     graphics  grDevices utils    
##  [8] datasets  methods   base     
## 
## other attached packages:
##  [1] RSNNS_0.4-9                     Rcpp_0.12.16                   
##  [3] rlang_0.2.0                     reshape2_1.4.3                 
##  [5] RColorBrewer_1.1-2              minerva_1.4.7                  
##  [7] markdown_0.7.7                  knitr_1.20                     
##  [9] glmnet_2.0-5                    foreach_1.4.3                  
## [11] Matrix_1.2-14                   gglasso_1.4                    
## [13] factoextra_1.0.5                DataExplorer_0.5.0             
## [15] corrplot_0.84                   autoEDA_1.0                    
## [17] rmarkdown_1.9                   party_1.3-0                    
## [19] strucchange_1.5-1               sandwich_2.3-4                 
## [21] zoo_1.8-1                       modeltools_0.2-21              
## [23] mvtnorm_1.0-7                   statmod_1.4.30                 
## [25] ROCR_1.0-7                      gplots_3.0.1                   
## [27] rjson_0.2.15                    RCurl_1.95-4.8                 
## [29] pROC_1.9.1                      mlbench_2.1-1                  
## [31] h2o_3.18.0.8                    ggbiplot_0.55                  
## [33] scales_0.5.0                    plyr_1.8.4                     
## [35] dplyr_0.7.4                     caret_6.0-76                   
## [37] ggplot2_2.2.1                   lattice_0.20-35                
## [39] bitops_1.0-6                    AppliedPredictiveModeling_1.1-6
## 
## loaded via a namespace (and not attached):
##  [1] nlme_3.1-137       rprojroot_1.2      backports_1.1.2   
##  [4] R6_2.2.2           rpart_4.1-13       KernSmooth_2.23-15
##  [7] lazyeval_0.2.0     colorspace_1.3-2   nnet_7.3-12       
## [10] gridExtra_2.3      curl_3.2           compiler_3.4.4    
## [13] formatR_1.5        labeling_0.3       caTools_1.17.1    
## [16] stringr_1.3.0      digest_0.6.15      foreign_0.8-70    
## [19] rio_0.5.10         pkgconfig_2.0.1    htmltools_0.3.6   
## [22] htmlwidgets_1.2    readxl_1.1.0       bindr_0.1.1       
## [25] jsonlite_1.5       gtools_3.5.0       ModelMetrics_1.1.0
## [28] car_3.0-0          CORElearn_1.52.1   magrittr_1.5      
## [31] munsell_0.4.3      abind_1.4-5        stringi_1.2.2     
## [34] multcomp_1.4-6     yaml_2.1.19        carData_3.0-1     
## [37] MASS_7.3-49        parallel_3.4.4     gdata_2.18.0      
## [40] ggrepel_0.7.0      forcats_0.3.0      haven_1.1.1       
## [43] splines_3.4.4      pillar_1.2.2       ggpubr_0.1.6      
## [46] igraph_1.0.1       codetools_0.2-15   glue_1.2.0        
## [49] evaluate_0.10      data.table_1.11.0  networkD3_0.4     
## [52] cellranger_1.1.0   purrr_0.2.4        gtable_0.2.0      
## [55] assertthat_0.2.0   openxlsx_4.0.17    coin_1.2-2        
## [58] e1071_1.6-8        class_7.3-14       survival_2.41-3   
## [61] tibble_1.4.2       iterators_1.0.8    bindrcpp_0.2.2    
## [64] cluster_2.0.7-1    TH.data_1.0-8
```


```r
getwd()
setwd("/media/disc/Megasync/R/regularization/group_lasso/code/")

# Load Packages
pkg.load <- c("autoEDA","bitops", "caret", "corrplot", "DataExplorer", "dplyr", "factoextra", "ggbiplot", "gglasso", "ggplot2", "glmnet", "h2o", "knitr", "markdown", "minerva", "mlbench","RColorBrewer", "RCurl", "reshape2", "rjson", "rmarkdown", "tools", "zoo", "rlang")



sapply(pkg.load, require, character.only = TRUE)


library(rmarkdown)
rmarkdown::render('main.Rmd')
# Document Updated last:
st.time.0.main <- Sys.time()

## Optional -- Install Packages
#library(devtools)
#install_github("vqv/ggbiplot")

# Load Packages
pkg.load <- c("autoEDA","bitops", "caret", "corrplot", "DataExplorer", "dplyr", "factoextra", "ggbiplot", "gglasso", "ggplot2", "glmnet", "h2o", "knitr", "markdown", "minerva", "mlbench","RColorBrewer", "RCurl", "reshape2", "rjson", "rmarkdown", "tools", "zoo", "rlang")



sapply(pkg.load, require, character.only = TRUE)


cacheData = FALSE
cachePlot = TRUE
fig_width_11 = 11
fig_height_8 = 8
# fig.width=fig_width_11, fig.height=fig_height_8
fig_width_7 = 7 
fig_height_5 = 5 
# fig.width=fig_width_7, fig.height=fig_height_5

knitr::opts_chunk$set(progress = TRUE, fig.width=fig_width_11, fig.height=fig_height_8, message = FALSE, warning = FALSE, fig.align = "center", tidy=TRUE, tidy.opts=list(blank=TRUE, width.cutoff=65) )
# cache = TRUE, cache.lazy = TRUE, 
# opts_knit$set(self.contained=FALSE)

ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/"
DATA.DIR <- paste(ROOT.DIR,"data", sep="")
CHART.DIR <- paste(ROOT.DIR, "charts", sep="")
CODE.DIR <- paste(ROOT.DIR, "code", sep="")
# Import Data
# data(Sonar)
# write.csv(Sonar, file = paste(DATA.DIR, "sonar.csv", sep="") )
#
Sonar <- read.csv(file = paste(DATA.DIR, "sonar.csv", sep="/") )
Sonar <- select(Sonar , -X)

# reomve rows with missing values
# Sonar_completeCase = Sonar[complete.cases(Sonar),]
# Sonar = Sonar[sample(nrow(Sonar)),]

# Assign Predictors and Response
y = Sonar$Class
X = Sonar[,-61]
# plot_missing(Sonar)
plot_bar(Sonar)
plot_histogram(Sonar)
plot_boxplot(Sonar, by="Class")

# plot_scatterplot(Sonar, by="Class")
dat_summary <- dataOverview(Sonar,outlierMethod = "tukey")


# Classification example:
power <-  predictivePower(x = Sonar,
                          y = "Class",
                          outcomeType = "automatic")

overview <- autoEDA(x=Sonar, y="Class")
ggplot(melt(Sonar[c(1:9,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 0.65, .1)) +
  facet_wrap(~variable)

ggplot(melt(Sonar[c(10:18,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 1, .25)) +
  facet_wrap(~variable)


ggplot(melt(Sonar[c(19:27,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 1, .25)) +
  facet_wrap(~variable)


ggplot(melt(Sonar[c(28:36,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 1, .25)) +
  facet_wrap(~variable)



ggplot(melt(Sonar[c(37:48,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 1, .25)) +
  facet_wrap(~variable)


ggplot(melt(Sonar[c(49:54,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 0.2, .05)) +
  facet_wrap(~variable)


ggplot(melt(Sonar[c(55:61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 0.05, .01)) +
  facet_wrap(~variable)



micValues <- mine(x=Sonar[,-61],
                  y = ifelse(Sonar$Class == "M", 1, 0), 
                  alpha=0.7)

# Analysis
M <- micValues$MIC
P <- cor(x=Sonar[,-61], y = ifelse(Sonar$Class == "M", 1, 0))

res <- data.frame(MIC = c(micValues$MIC))
# res <- data.frame(M)
rownames(res) <- rownames(micValues$MIC)
res$MIC_Rank <- nrow(res) - rank(res$MIC, ties.method="first") + 1
res$Pearson <- P
res$Pearson_Rank <- nrow(res) - rank(abs(res$Pearson), ties.method="first") + 1
res <- res[order(res$MIC_Rank),]
head(res, n=10)
tail(res, n=10)
# Plot
# png("MIC_vs_Pearson.png", width=7.5, height=3.5, res=400, units="in", type="cairo")
op <- par(mfrow=c(1,2), mar=c(4,4,1,1))
plot(MIC ~ abs(Pearson), res, pch=21,  col=4, bg=5)
plot(MIC_Rank ~ Pearson_Rank, res, pch=21, col=4, bg=5)
par(op)
# dev.off()

plot_correlation(Sonar )

# Dissimilarity matrix
d = dist(t(X), method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 = hclust(d, method = "complete" )

# Apply Elbow rule
fviz_nbclust(X, FUN = hcut, method = "wss") # It seems 4 - 6 groups may be appropriate

# Plot dendrogram with 5 selected groups
plot(hc1, cex = 0.6, hang = -1)
rect.hclust(hc1, k = 5, border = 2:5)
# source('/media/disc/Megasync/R/regularization/group_lasso/code/grouped_lasso.Rmd')

#spin_child('/media/disc/Megasync/R/regularization/group_lasso/code/spinner_test.R')
{{knitr::spin_child('/media/disc/Megasync/R/regularization/group_lasso/code/deep_learn_h20_roxy.R')}}
st.time.0.main

end.time.0.main <- Sys.time()
end.time.0.main 

end.time.0.main   - st.time.0.main  
getwd()

list.files(getwd())
sessionInfo()



# Download packages that H2O depends on.
pkgs <- c("pROC", "RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
# Load libs
pkg.load <- c("AppliedPredictiveModeling", "bitops", "caret", "dplyr", "ggbiplot", "h2o", "mlbench", "pROC", "RCurl", "rjson", "ROCR", "statmod", "tools")
sapply(pkg.load, require, character.only = TRUE)
source("/media/disc/Megasync/R/regularization/group_lasso/code/helper_h2o.R")


ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/"
DATA.DIR <- paste(ROOT.DIR,"data",sep="")
CODE.DIR <- paste(ROOT.DIR, "code", sep="")
seed = 77
set.seed(seed)
h2o <- h2o.init(nthreads = -1, max_mem_size = "14g" )
# data(Sonar)
# write.csv(Sonar, file = paste(DATA.DIR, "sonar_dl.csv", sep="/"), row.names=FALSE )
Sonar <- read.csv(file = paste(DATA.DIR, "sonar_dl.csv", sep="/"))
#
pathToData <- paste0(normalizePath(DATA.DIR), "/sonar_dl.csv")
# sonar.hex = h2o.importFile("/media/disc/Megasync/R/regularization/group_lasso/data/sonar_dl.csv")
sonar.hex <- h2o.importFile(path = pathToData, destination_frame = "sonar.hex")
# 
sonar.df = as.data.frame(sonar.hex)
#
# Create the feature vectors by dropping the classifications
classVariableIndex = 61
sonar.features = sonar.df[, -classVariableIndex]
# Rename the classification varible to something meaningful
names(sonar.df)[classVariableIndex] = "Class"
sonar.split = h2o.splitFrame(data = sonar.hex,
                             ratios = 0.8,
                             seed = seed)
#
# Create named varaibles for the traning and test data
parts <- sonar.split[[1]]
parts <- h2o.splitFrame(parts, 1.0/6.0)
sonar.valid <- parts[[1]]
sonar.train <- parts[[2]]
sonar.test = sonar.split[[2]]
#
# Create R data frames from the H2O data frames
sonar.train.df = as.data.frame(sonar.train)
names(sonar.train.df)[classVariableIndex] = "Class"
sonar.test.df = as.data.frame(sonar.test)
names(sonar.test.df)[classVariableIndex] = "Class"
transparentTheme(trans = 0.5)
featurePlot <- featurePlot(
  x = sonar.features,
  y = sonar.df$Class,
  plot = "pairs",
  pscales = FALSE,
  cex = 0.05,
  auto.key = list(columns = 2) )
featurePlot
# Use principal components to reduce the dimensionality of the features.
sonar.princomp = prcomp(sonar.features,  
                        center = TRUE,
                        scale = TRUE,
                        retx = TRUE)
sonar.princomp.7 = prcomp(sonar.features, 
                        center = TRUE,
                        scale = TRUE,
                        tol=.4, 
                        retx = TRUE)
summary(sonar.princomp.7)
# PCA TRELLIS PLOT 
transparentTheme(trans = 0.3)
numberOfPrincipalComponentsToPlot = 3
featurePlot(
  x = sonar.princomp$x[, 1:numberOfPrincipalComponentsToPlot],
  y = sonar.df$Class,
  plot = "pairs",
  pscales = FALSE,
  auto.key = list(columns = 2)
)
g <- ggbiplot(
  sonar.princomp,
  groups = sonar.df$Class,
  var.axes = FALSE,
  #
  ellipse = TRUE
  )
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal',
               legend.position = 'top')
print(g)
run.DTREE = 0
if(run.DTREE == 1){
  time.dtree.fit <- Sys.time()
  fit.d.tree = train(Class ~ .,  sonar.train.df, method="ctree")
  time.dtree.end <- Sys.time()
  time.dtree.diff <- time.dtree.end - time.dtree.fit
  
  saveRDS(time.dtree.diff, file = paste(DATA.DIR, "time.dtree.diff.rds", sep="/"))
  save(fit.d.tree,
       file = paste(DATA.DIR, "fit.d.tree.Rdata", sep="/"))
  saveRDS(fit.d.tree,
          file = paste(DATA.DIR, "fit.d.tree.rds", sep="/") )  
}
#
time.dtree.diff <- readRDS(file = paste(DATA.DIR, "time.dtree.diff.rds", sep="/"))
# fit.cv.ridge <- readRDS(file = paste(DATA.DIR, "fit.cv.ridge.rds", sep="/"))
load(file = paste(DATA.DIR, "fit.d.tree.Rdata", sep="/"))
plot(fit.d.tree$finalModel)
pred.d.tree = predict(fit.d.tree, sonar.test.df[, -classVariableIndex])
res.d.tree <- caret::confusionMatrix(pred.d.tree, sonar.test.df$Class)
pred.d.tree.probabilities=predict(fit.d.tree, sonar.test.df[, -classVariableIndex], type="prob")
d.tree.roc=roc(sonar.test.df$Class, pred.d.tree.probabilities$M)
# Area under the curve: 0.8433
plot(d.tree.roc)
model.gbm.exp <- h2o.experiment(h2o.gbm)
m.gbm0 <- h2o.gbm(1:60, 61, sonar.train,
                  nfolds = 0 ,model_id = "GBM_default")
m.gbm0

# Random forest. A good ensemble learner.
model.rf.exp <- h2o.experiment(h2o.randomForest)
m.rf0 <- h2o.randomForest(1:60, 61,
                              sonar.train, 
                              nfolds = 0, 
                              model_id = "RF_default")
summary(m.rf0)
h2o.performance(m.rf0, sonar.test)



res <- compareModels(c(m.gbm0, m.rf0), sonar.test)
round(res[,"AUC",], 3)
#
compareModels <- function(models, test, labels = NULL){
  #Use model IDs as default labels, if not given  
  if(is.null(labels)){
    labels <- lapply(models, function(m) m@model_id)
  }
  
  res <- sapply(models, function (m){
    mcmsT <- m@model$training_metrics@metrics$max_criteria_and_metric_scores
    mcmsV <- m@model$validation_metrics@metrics$max_criteria_and_metric_scores
    maix <- which(mcmsT$metric=="max accuracy")  #4 (at the time of writing)
    th <- mean(mcmsT[maix, 'threshold'],  mcmsV[maix, 'threshold'] )
    
    pf <- h2o.performance(m, test)
    tms <- pf@metrics$thresholds_and_metric_scores
    ix <- apply(outer(th, tms$threshold, "<="), 1, sum)
    if(ix < 1)ix <- 1  #Use first entry if less than all of them
    
    matrix(c(
      h2o.auc(m, TRUE, TRUE), pf@metrics$AUC,
      mcmsT[maix, 'value'], mcmsV[maix, 'value'], tms[ix, 'accuracy'],
      h2o.logloss(m, TRUE, TRUE), pf@metrics$logloss,
      h2o.mse(m, TRUE, TRUE), pf@metrics$MSE
    ), ncol = 4)
  }, simplify = "array")
  
  dimnames(res) <- list(
    c("train","valid","test"),
    c("AUC","Accuracy","logloss", "MSE"),
    labels
  )
  
  res
}
h2o.experiment(h2o.glm, list(family = "binomial"))


fTable1 <- kable(forec.table.h6.p1,digits=0 , format = "markdown", padding = 2, caption="Forecast results for candidate models")
fTable2 <- kable(forec.table.h6.p2,digits=0 , format = "markdown", padding = 2, caption="Forecast results for candidate models")
print(fTable1)
print(fTable2)
dl.fit <- h2o.experiment(h2o.deeplearning)
```




# Chapter 2





------------------------------------------------------      
# Abstract
        
In 2006, Yuan and Lin introduced the group lasso in order to allow predefined groups of covariates to be selected into or out of a model together, so that all the members of a particular group are either included or not included. While there are many settings in which this is useful, perhaps the most obvious is when levels of a categorical variable are coded as a collection of binary covariates. In this case, it often doesn't make sense to include only a few levels of the covariate; the group lasso can ensure that all the variables encoding the categorical covariate are either included or excluded from the model together.                          
                   
The idea behind this report is to not only delve into the mathematical details of the Grouped LASSO, but also simplify the details so that the average reader can walk away with an approximate understanding of this particular derivation of the LASSO. The traditional LASSO is amenable to data sets where sparsity is more likely to uncover the "true" model, but a downfall of this method is its inability to perform well when the design matrix contains much multicollinearity. In such an event we often find the L1 regularization to not only produce a sub-standard model, but it will also fail at obtaining maximal sparsity or any as is seen in the case of the Sonar data set we introduce in a later section. Furthermore, we make use of the Sonar data set to illustrate how the application of the Grouped LASSO can ameliorate the formerly mentioned sparsity issue in the light of multicollinearity.        
            
The mathematics and formulae behind the Grouped LASSO are discussed, but due to it's complex nature and even more complex solution path we have added a small example to illustrate the inner workings of this derivation. We also discuss how to group your predictors based on correlation as well as the interesting result that sparsity is not only obtained, but the algorithm has an ability to do so by predictor within a group and not by eliminating entire groups. We do, however, avoid discussions about the variants of the Group LASSO: Sparse Group LASSO, Overlap Group LASSO, Sparse GAM with Group LASSO etc.  

# The Grouped LASSO

There are many regression problems in which the covariates have a natural group structure, and it is desirable to have all coefficients within a group become nonzero (or zero) simultaneously. The various forms of group lasso penalty are designed for such situations. We first define the group lasso and then develop this and other motivating examples.                           
The setup is as follows:          
         
*There are $J$ groups where $j = 1,...,J$
*The vector $Z_j \in \mathbb{R}^{P_j}$ represents the covariates in group j       
      
The Goal:        
     
To predict a real-values response $Y \in \mathbb{R}$ based on the collection of covariates in our $J$ groups $(Z_1,...,Z_j)$ 

##Defining the Linear Model
          
The linear model can be defined as:     
      
$$\mathbb{E}(Y|Z) = \theta_0 + \sum_{j=1}^JZ_j^T\theta_j,  \ where \  \theta_j \in \mathbb{R}^{P_j} $$        
            
Note: $\theta_j$ represents a group of $p_j$ regression coefficients.     
        
##Defining The Convex Problem         
         
Given a collection of $N$ samples $\{(y_i,z_{i1},z_{i2},...,z_{i,J})\}^N_{i=1}$ the Group LASSO solves the following covex problem:        
        
$$\underset{\theta_0 \in \mathbb{R}, \theta_j \in \mathbb{R^{p_j}} }{\operatorname{minimize}} \left\{\frac{1}{2}\sum_{i=1}^N(y_i - \theta_0 - \sum_{j=1}^Jz_{ij}^T\theta_j)^2 + \lambda\sum_{j=1}^J\|\theta_j\|_2\right\}$$ 
Where $\|\theta_j\|_2$ is the Euclidean norm of the vector $\theta_j$ and the following hold true:          

* depending on $\lambda \ge 0$, either the entire vector $\hat{\theta_j}$ will be zero, or all its elements will be nonzero.                
* when $p_j = 1$, then we have $\|\theta_j\|_2 = |\theta_j|$, so if all the groups are singletons i.e. every group represents a single predictor, than the optimization problem reduces to the ordinary LASSO.       
* All groups are equally penalized, a choice which leads larger groups to be more likely to be selected. In their original proposal, Yuan and Lin (2006)[1] recommended weighting the penalties for each group according to their size, by a factor $\sqrt{p_j}$. One could also argue for a factor of $\|\mathbb{Z}_j\|_F$ where the matrices are not orthonormal.                 
       
Here we compare the constraint region for the Group LASSO (left) to that of the LASSO in $\mathbb{R}^3$. We see that the Group LASSO shares attributes of both the $l_2$ and $l_1$ balls:                
             
```
![My Folder](Group LASSO.JPG) 
```                 
             
# Multi-Level Sparsity: Sparse Group LASSO        
         
I would be remiss if I didn't, at the very least, discuss the Sparse Group LASSO, which takes the Group LASSO a step further by not only imposing sparsity on the groups, but also selects which coefficients are non-zero within the groups. From a technical standpoint this is vital considering the core uncertainty in this style of problem if that of determining the groups. If you have selected your groups to include, even just one, important variable(s) then this coefficient would be shrunk to zero along with all other coefficients in said group, however with the advantages of using the Sparse Group LASSO there is a strong chance that "important" coefficients within zeroed groups may be recovered in the final model.      
        
In order to achieve within group sparsity, we augment the basic Group LASSO with an additional $l_1$-penalty, leading to the convex program:        
        
$$\underset{\{\theta_j \in \mathbb{R}^{p_j}\}}{\operatorname{minimize}} \left\{\frac{1}{2}\|\mathbf{y} - \sum_{j=1}^J\mathbf(Z)_j\theta_j\|_2^2 + \lambda \sum_{j=1}^J[(1-\alpha)\|\theta_j\|_2 + \alpha\|\theta_j\|_1] \right\}$$
                 
with $\alpha \in [0,1]$. Much like the Elastic Net, the parameter $\alpha$ creates a bridge between the Group LASSO ($\alpha = 0$) and the LASSO ($\alpha = 1$). Below is the image that contrasts the Group LASSO constraint region with that of the Sparse Group LASSO for the case in $\mathbb{R}^3$:    
    
```
![My Folder](Sparse Group LASSO.JPG)        
```

*Note: in the two horizontal planes the constraint region resembles that of the elastic net being more rounded than angular.*      
                 
# The Sonar Data Set                  
              
## Description      
         
This is the data set used by Gorman and Sejnowski in their study of the classification of sonar signals using a neural network [2]. The task is to train a network to discriminate between sonar signals bounced off a metal cylinder and those bounced off a roughly cylindrical rock.        
         
Each pattern is a set of 60 numbers in the range 0.0 to 1.0 [3]. Each number represents the energy within a particular frequency band, integrated over a certain period of time. The integration aperture for higher frequencies occur later in time, since these frequencies are transmitted later during the chirp.         
            
The label associated with each record contains the letter "R" if the object is a rock and "M" if it is a mine (metal cylinder). The numbers in the labels are in increasing order of aspect angle, but they do not encode the angle directly.          
          
Below is a preview of the data set:                         
                

       
## Why Use A Group LASSO?         
       
The primary issue with the Sonar data is the eggregious multicollinearity present. We display this with a correlation heat map below:              
<img src="main_files/figure-html/unnamed-chunk-8-1.png" style="display: block; margin: auto;" />
             
As we can see from the above correlation heat map, we have thick clusters of predictors that are highly correlated with one another. This is an issue when trying to predict the response with the LASSO. Our results would typically contain high variability and almost no sparsity is obtained, but this is almost single handedly ameliorated with the Group LASSO.       
          
## How to Determine Groups?            
            
There are various clustering algorithms that can be utilized when forming groups: The method that seemed to work the best is Hierarchical Clustering with a Euclidean distance and a Complete linkage that I determined through trial and error, however I did investigate a contemporary clustering algorithm called DBSCAN (Density Based Clustering of Applications with Noise). Because this report is focused on the Group LASSO I will ommit the more techinical details, but I will state that it did not perform well because the groups are too sparse in the PC1/PC2 dimension.        
          
Below we apply the "elbow rule" to the Scree Plot and highlight the 5 selected groups on the Dendrogram:                       

<img src="main_files/figure-html/unnamed-chunk-9-1.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-9-2.png" style="display: block; margin: auto;" />

### Train Test Split         
        
We split the data set into a 90/10 train-test split. The main reason for this type of split is purely because the number of samples is disproportionate to the number of observations. The training set is left containing 3 samples for every predictor. This is far below the recomended rule of thumb of 10 (for Regression problems), albeit an area where the LASSO algorithm shines.       
           


## Model's Implimented                    
              
We fit and compare the following models:            

**1) Multi-Layer Perceptron (MLP)**       
       
The first model I chose simply because the developers of the Sonar data set (Gorman & Sejnowski 1988)[2] fit a MLP model to this data with perfect prediction accuracy. I have very little doubt that the training took an immense amount of time with many iterations to the number of layers and included neurons. I made use of the SNNS (Stuttgart Neural Network Simulator) package in R. We tuned this model using a tuning grid for three layers with three settings of 10, 20, and 30 hidden neurons. Due to time constraints a larger tuning grid was unable to be explored.          
       
**2) LASSO**            
        
This model was fit using 10 fold 5 repeated cross validation strategy within the framework of the "Caret" package. I did make use of a tuning grid for $\lambda$.        
         
**3) Elastic Net**                      
          
This model was fit using 10 fold 5 repeated cross validation strategy within the framework of the "Caret" package. I did make use of a tuning grid for $\alpha$ and $\lambda$.
              
**4) Group LASSO**             
         
This model was fit using a cross validation function within the "gglasso" package with 5 folds as an option to balance the CV settings of the LASSO/Elastic Net combination with the Group LASSO/Group Ridge combination for comparability.        
          
**5) Group Ridge**            
         
This model was fit using a cross validation function within the "gglasso" package with 5 folds as an option to balance the CV settings of the LASSO/Elastic Net combination with the Group LASSO/Group Ridge combination for comparability.                  
          
*The comparison will be made based on test sample prediction accuracy, time elapsed, and model sparsity.*              
            
## Model Comparison           
         



              
### Time Comparison (GROUPED LASSO doc)  
             
Compare model results:              

            
         
Model           |Accuracy                      | Run Time            |No. of Predictors|
----------------|------------------------------|---------------------|-----------------|
MLP             |0.55   |6.85   |60|
LASSO           |1 |13.94     |52|
Elastic Net     |0.7  |3.08    |60|
Group LASSO     |0.7|41.77|46   |
Group Ridge     |0.7|37.62|60   |   
Decision Tree   | 0.8  | 4.64 | 6   |
      
           
**Analysis**             
          
The outright winner if run time is the exclusive concern would typically be the Elastic Net and the worst run time is the Group LASSO.
      
### Sparsity Comparison    
          
*We omit the MLP and the Group Ridge from this discussion since they are not sparse models to begin with*   
           
**LASSO**           
            

```
## png 
##   2
```
          
The accuracy of the LASSO model is highest when there is no coefficient shrinkage. This is to be expected from the multicollinearity issue we had previously assessed and described above.         
          
**Group LASSO**           
            
<img src="main_files/figure-html/unnamed-chunk-12-1.png" style="display: block; margin: auto;" /><img src="main_files/figure-html/unnamed-chunk-12-2.png" style="display: block; margin: auto;" />
           
Here we see that not only was there shrinkage but there were 10 coefficients shrunk to zero, however these were all coefficients that belonged to Group 1, but did not encompass the full membership of Group 1 leading me to believe that the package Glasso impliments the Sparse Group LASSO and not the Group LASSO singularly.                        
                
**Elastic NET**           
            
<img src="main_files/figure-html/unnamed-chunk-13-1.png" style="display: block; margin: auto;" />
          
Once again we see a similar plot to that of the LASSO where all the repeated CV attempts achieved highest accuracy with no parameter shrinkage.                  
          
### Prediction Accuracy          
         
**MLP**         
     

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  M  R
##          M 11  9
##          R  0  0
##                                           
##                Accuracy : 0.55            
##                  95% CI : (0.3153, 0.7694)
##     No Information Rate : 0.55            
##     P-Value [Acc > NIR] : 0.591361        
##                                           
##                   Kappa : 0               
##  Mcnemar's Test P-Value : 0.007661        
##                                           
##             Sensitivity : 1.00            
##             Specificity : 0.00            
##          Pos Pred Value : 0.55            
##          Neg Pred Value :  NaN            
##              Prevalence : 0.55            
##          Detection Rate : 0.55            
##    Detection Prevalence : 1.00            
##       Balanced Accuracy : 0.50            
##                                           
##        'Positive' Class : M               
## 
```


**LASSO**         
     

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  M  R
##          M 11  0
##          R  0  9
##                                      
##                Accuracy : 1          
##                  95% CI : (0.8316, 1)
##     No Information Rate : 0.55       
##     P-Value [Acc > NIR] : 0.000006416
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
##                                      
##             Sensitivity : 1.00       
##             Specificity : 1.00       
##          Pos Pred Value : 1.00       
##          Neg Pred Value : 1.00       
##              Prevalence : 0.55       
##          Detection Rate : 0.55       
##    Detection Prevalence : 0.55       
##       Balanced Accuracy : 1.00       
##                                      
##        'Positive' Class : M          
## 
```


**Eslastic Net**         
     

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction M R
##          M 8 3
##          R 3 6
##                                           
##                Accuracy : 0.7             
##                  95% CI : (0.4572, 0.8811)
##     No Information Rate : 0.55            
##     P-Value [Acc > NIR] : 0.1299          
##                                           
##                   Kappa : 0.3939          
##  Mcnemar's Test P-Value : 1.0000          
##                                           
##             Sensitivity : 0.7273          
##             Specificity : 0.6667          
##          Pos Pred Value : 0.7273          
##          Neg Pred Value : 0.6667          
##              Prevalence : 0.5500          
##          Detection Rate : 0.4000          
##    Detection Prevalence : 0.5500          
##       Balanced Accuracy : 0.6970          
##                                           
##        'Positive' Class : M               
## 
```


**Group LASSO**         
     

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction M R
##          M 8 3
##          R 3 6
##                                           
##                Accuracy : 0.7             
##                  95% CI : (0.4572, 0.8811)
##     No Information Rate : 0.55            
##     P-Value [Acc > NIR] : 0.1299          
##                                           
##                   Kappa : 0.3939          
##  Mcnemar's Test P-Value : 1.0000          
##                                           
##             Sensitivity : 0.7273          
##             Specificity : 0.6667          
##          Pos Pred Value : 0.7273          
##          Neg Pred Value : 0.6667          
##              Prevalence : 0.5500          
##          Detection Rate : 0.4000          
##    Detection Prevalence : 0.5500          
##       Balanced Accuracy : 0.6970          
##                                           
##        'Positive' Class : M               
## 
```


**Ridge Group**         
     

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction M R
##          M 8 3
##          R 3 6
##                                           
##                Accuracy : 0.7             
##                  95% CI : (0.4572, 0.8811)
##     No Information Rate : 0.55            
##     P-Value [Acc > NIR] : 0.1299          
##                                           
##                   Kappa : 0.3939          
##  Mcnemar's Test P-Value : 1.0000          
##                                           
##             Sensitivity : 0.7273          
##             Specificity : 0.6667          
##          Pos Pred Value : 0.7273          
##          Neg Pred Value : 0.6667          
##              Prevalence : 0.5500          
##          Detection Rate : 0.4000          
##    Detection Prevalence : 0.5500          
##       Balanced Accuracy : 0.6970          
##                                           
##        'Positive' Class : M               
## 
```
           
# Conclusion         
       
The Group LASSO not only achieves sparsity, but it does so with the highest accuracy, however one should be cautious of the time taken to run since this algorithm becomes increasingly slow with the increase in predictors and is further slowed down when running combining methods like Cross Validation.
           
# References     
         
[1] Yuan, M. and Lin, Y. (2006) "Model Selection and Estimation in Regression with Grouped Variables". *J. R. Statist. Soc.* B, 68, 49-67            
[2] Hastie. H, Tibshirani. R, and Wainwright, M. (2016) "Statistical Learning With Sparsity: The Lasso and Generalizations". *CRC Press*          
[3] Gorman, R. P., and Sejnowski, T. J. (1988). "Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets" in Neural Networks, Vol. 1, pp. 75-89.               
[4] Newman, D.J. & Hettich, S. & Blake, C.L. & Merz, C.J. (1998). UCI Repository of machine learning databases [http://www.ics.uci.edu/~mlearn/MLRepository.html]. Irvine, CA: University of California, Department of Information and Computer Science.             
           

## Document Settings  

* Document date of last update



```r
stTime
```

```
## [1] "2018-05-09 12:23:01 EDT"
```

```r
endTime <- Sys.time()
endTime
```

```
## [1] "2018-05-09 12:23:03 EDT"
```

```r
endTime - stTime
```

```
## Time difference of 1.902416 secs
```
## R Code for Report



```r
getwd()
setwd("/media/disc/Megasync/R/regularization/group_lasso/code/")

# Load Packages
pkg.load <- c("autoEDA","bitops", "caret", "corrplot", "DataExplorer", "dplyr", "factoextra", "ggbiplot", "gglasso", "ggplot2", "glmnet", "h2o", "knitr", "markdown", "minerva", "mlbench","RColorBrewer", "RCurl", "reshape2", "rjson", "rmarkdown", "tools", "zoo", "rlang")



sapply(pkg.load, require, character.only = TRUE)


library(rmarkdown)
rmarkdown::render('main.Rmd')
# Document Updated last:
st.time.0.main <- Sys.time()

## Optional -- Install Packages
#library(devtools)
#install_github("vqv/ggbiplot")

# Load Packages
pkg.load <- c("autoEDA","bitops", "caret", "corrplot", "DataExplorer", "dplyr", "factoextra", "ggbiplot", "gglasso", "ggplot2", "glmnet", "h2o", "knitr", "markdown", "minerva", "mlbench","RColorBrewer", "RCurl", "reshape2", "rjson", "rmarkdown", "tools", "zoo", "rlang")



sapply(pkg.load, require, character.only = TRUE)


cacheData = FALSE
cachePlot = TRUE
fig_width_11 = 11
fig_height_8 = 8
# fig.width=fig_width_11, fig.height=fig_height_8
fig_width_7 = 7 
fig_height_5 = 5 
# fig.width=fig_width_7, fig.height=fig_height_5

knitr::opts_chunk$set(progress = TRUE, fig.width=fig_width_11, fig.height=fig_height_8, message = FALSE, warning = FALSE, fig.align = "center", tidy=TRUE, tidy.opts=list(blank=TRUE, width.cutoff=65) )
# cache = TRUE, cache.lazy = TRUE, 
# opts_knit$set(self.contained=FALSE)

ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/"
DATA.DIR <- paste(ROOT.DIR,"data", sep="")
CHART.DIR <- paste(ROOT.DIR, "charts", sep="")
CODE.DIR <- paste(ROOT.DIR, "code", sep="")
# Import Data
# data(Sonar)
# write.csv(Sonar, file = paste(DATA.DIR, "sonar.csv", sep="") )
#
Sonar <- read.csv(file = paste(DATA.DIR, "sonar.csv", sep="/") )
Sonar <- select(Sonar , -X)

# reomve rows with missing values
# Sonar_completeCase = Sonar[complete.cases(Sonar),]
# Sonar = Sonar[sample(nrow(Sonar)),]

# Assign Predictors and Response
y = Sonar$Class
X = Sonar[,-61]
# plot_missing(Sonar)
plot_bar(Sonar)
plot_histogram(Sonar)
plot_boxplot(Sonar, by="Class")

# plot_scatterplot(Sonar, by="Class")
dat_summary <- dataOverview(Sonar,outlierMethod = "tukey")


# Classification example:
power <-  predictivePower(x = Sonar,
                          y = "Class",
                          outcomeType = "automatic")

overview <- autoEDA(x=Sonar, y="Class")
ggplot(melt(Sonar[c(1:9,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 0.65, .1)) +
  facet_wrap(~variable)

ggplot(melt(Sonar[c(10:18,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 1, .25)) +
  facet_wrap(~variable)


ggplot(melt(Sonar[c(19:27,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 1, .25)) +
  facet_wrap(~variable)


ggplot(melt(Sonar[c(28:36,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 1, .25)) +
  facet_wrap(~variable)



ggplot(melt(Sonar[c(37:48,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 1, .25)) +
  facet_wrap(~variable)


ggplot(melt(Sonar[c(49:54,61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 0.2, .05)) +
  facet_wrap(~variable)


ggplot(melt(Sonar[c(55:61)], id.vars = "Class"), 
       aes(x=value,colour=Class)) +  
  geom_freqpoly(aes(y = ..density..)) +
  scale_x_continuous(breaks = seq(0, 0.05, .01)) +
  facet_wrap(~variable)



micValues <- mine(x=Sonar[,-61],
                  y = ifelse(Sonar$Class == "M", 1, 0), 
                  alpha=0.7)

# Analysis
M <- micValues$MIC
P <- cor(x=Sonar[,-61], y = ifelse(Sonar$Class == "M", 1, 0))

res <- data.frame(MIC = c(micValues$MIC))
# res <- data.frame(M)
rownames(res) <- rownames(micValues$MIC)
res$MIC_Rank <- nrow(res) - rank(res$MIC, ties.method="first") + 1
res$Pearson <- P
res$Pearson_Rank <- nrow(res) - rank(abs(res$Pearson), ties.method="first") + 1
res <- res[order(res$MIC_Rank),]
head(res, n=10)
tail(res, n=10)
# Plot
# png("MIC_vs_Pearson.png", width=7.5, height=3.5, res=400, units="in", type="cairo")
op <- par(mfrow=c(1,2), mar=c(4,4,1,1))
plot(MIC ~ abs(Pearson), res, pch=21,  col=4, bg=5)
plot(MIC_Rank ~ Pearson_Rank, res, pch=21, col=4, bg=5)
par(op)
# dev.off()

plot_correlation(Sonar )

# Dissimilarity matrix
d = dist(t(X), method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 = hclust(d, method = "complete" )

# Apply Elbow rule
fviz_nbclust(X, FUN = hcut, method = "wss") # It seems 4 - 6 groups may be appropriate

# Plot dendrogram with 5 selected groups
plot(hc1, cex = 0.6, hang = -1)
rect.hclust(hc1, k = 5, border = 2:5)
# source('/media/disc/Megasync/R/regularization/group_lasso/code/grouped_lasso.Rmd')

#spin_child('/media/disc/Megasync/R/regularization/group_lasso/code/spinner_test.R')
{{knitr::spin_child('/media/disc/Megasync/R/regularization/group_lasso/code/deep_learn_h20_roxy.R')}}
st.time.0.main

end.time.0.main <- Sys.time()
end.time.0.main 

end.time.0.main   - st.time.0.main  
getwd()

list.files(getwd())
sessionInfo()



# Download packages that H2O depends on.
pkgs <- c("pROC", "RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
# Load libs
pkg.load <- c("AppliedPredictiveModeling", "bitops", "caret", "dplyr", "ggbiplot", "h2o", "mlbench", "pROC", "RCurl", "rjson", "ROCR", "statmod", "tools")
sapply(pkg.load, require, character.only = TRUE)
source("/media/disc/Megasync/R/regularization/group_lasso/code/helper_h2o.R")


ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/"
DATA.DIR <- paste(ROOT.DIR,"data",sep="")
CODE.DIR <- paste(ROOT.DIR, "code", sep="")
seed = 77
set.seed(seed)
h2o <- h2o.init(nthreads = -1, max_mem_size = "14g" )
# data(Sonar)
# write.csv(Sonar, file = paste(DATA.DIR, "sonar_dl.csv", sep="/"), row.names=FALSE )
Sonar <- read.csv(file = paste(DATA.DIR, "sonar_dl.csv", sep="/"))
#
pathToData <- paste0(normalizePath(DATA.DIR), "/sonar_dl.csv")
# sonar.hex = h2o.importFile("/media/disc/Megasync/R/regularization/group_lasso/data/sonar_dl.csv")
sonar.hex <- h2o.importFile(path = pathToData, destination_frame = "sonar.hex")
# 
sonar.df = as.data.frame(sonar.hex)
#
# Create the feature vectors by dropping the classifications
classVariableIndex = 61
sonar.features = sonar.df[, -classVariableIndex]
# Rename the classification varible to something meaningful
names(sonar.df)[classVariableIndex] = "Class"
sonar.split = h2o.splitFrame(data = sonar.hex,
                             ratios = 0.8,
                             seed = seed)
#
# Create named varaibles for the traning and test data
parts <- sonar.split[[1]]
parts <- h2o.splitFrame(parts, 1.0/6.0)
sonar.valid <- parts[[1]]
sonar.train <- parts[[2]]
sonar.test = sonar.split[[2]]
#
# Create R data frames from the H2O data frames
sonar.train.df = as.data.frame(sonar.train)
names(sonar.train.df)[classVariableIndex] = "Class"
sonar.test.df = as.data.frame(sonar.test)
names(sonar.test.df)[classVariableIndex] = "Class"
transparentTheme(trans = 0.5)
featurePlot <- featurePlot(
  x = sonar.features,
  y = sonar.df$Class,
  plot = "pairs",
  pscales = FALSE,
  cex = 0.05,
  auto.key = list(columns = 2) )
featurePlot
# Use principal components to reduce the dimensionality of the features.
sonar.princomp = prcomp(sonar.features,  
                        center = TRUE,
                        scale = TRUE,
                        retx = TRUE)
sonar.princomp.7 = prcomp(sonar.features, 
                        center = TRUE,
                        scale = TRUE,
                        tol=.4, 
                        retx = TRUE)
summary(sonar.princomp.7)
# PCA TRELLIS PLOT 
transparentTheme(trans = 0.3)
numberOfPrincipalComponentsToPlot = 3
featurePlot(
  x = sonar.princomp$x[, 1:numberOfPrincipalComponentsToPlot],
  y = sonar.df$Class,
  plot = "pairs",
  pscales = FALSE,
  auto.key = list(columns = 2)
)
g <- ggbiplot(
  sonar.princomp,
  groups = sonar.df$Class,
  var.axes = FALSE,
  #
  ellipse = TRUE
  )
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal',
               legend.position = 'top')
print(g)
run.DTREE = 0
if(run.DTREE == 1){
  time.dtree.fit <- Sys.time()
  fit.d.tree = train(Class ~ .,  sonar.train.df, method="ctree")
  time.dtree.end <- Sys.time()
  time.dtree.diff <- time.dtree.end - time.dtree.fit
  
  saveRDS(time.dtree.diff, file = paste(DATA.DIR, "time.dtree.diff.rds", sep="/"))
  save(fit.d.tree,
       file = paste(DATA.DIR, "fit.d.tree.Rdata", sep="/"))
  saveRDS(fit.d.tree,
          file = paste(DATA.DIR, "fit.d.tree.rds", sep="/") )  
}
#
time.dtree.diff <- readRDS(file = paste(DATA.DIR, "time.dtree.diff.rds", sep="/"))
# fit.cv.ridge <- readRDS(file = paste(DATA.DIR, "fit.cv.ridge.rds", sep="/"))
load(file = paste(DATA.DIR, "fit.d.tree.Rdata", sep="/"))
plot(fit.d.tree$finalModel)
pred.d.tree = predict(fit.d.tree, sonar.test.df[, -classVariableIndex])
res.d.tree <- caret::confusionMatrix(pred.d.tree, sonar.test.df$Class)
pred.d.tree.probabilities=predict(fit.d.tree, sonar.test.df[, -classVariableIndex], type="prob")
d.tree.roc=roc(sonar.test.df$Class, pred.d.tree.probabilities$M)
# Area under the curve: 0.8433
plot(d.tree.roc)
model.gbm.exp <- h2o.experiment(h2o.gbm)
m.gbm0 <- h2o.gbm(1:60, 61, sonar.train,
                  nfolds = 0 ,model_id = "GBM_default")
m.gbm0

# Random forest. A good ensemble learner.
model.rf.exp <- h2o.experiment(h2o.randomForest)
m.rf0 <- h2o.randomForest(1:60, 61,
                              sonar.train, 
                              nfolds = 0, 
                              model_id = "RF_default")
summary(m.rf0)
h2o.performance(m.rf0, sonar.test)



res <- compareModels(c(m.gbm0, m.rf0), sonar.test)
round(res[,"AUC",], 3)
#
compareModels <- function(models, test, labels = NULL){
  #Use model IDs as default labels, if not given  
  if(is.null(labels)){
    labels <- lapply(models, function(m) m@model_id)
  }
  
  res <- sapply(models, function (m){
    mcmsT <- m@model$training_metrics@metrics$max_criteria_and_metric_scores
    mcmsV <- m@model$validation_metrics@metrics$max_criteria_and_metric_scores
    maix <- which(mcmsT$metric=="max accuracy")  #4 (at the time of writing)
    th <- mean(mcmsT[maix, 'threshold'],  mcmsV[maix, 'threshold'] )
    
    pf <- h2o.performance(m, test)
    tms <- pf@metrics$thresholds_and_metric_scores
    ix <- apply(outer(th, tms$threshold, "<="), 1, sum)
    if(ix < 1)ix <- 1  #Use first entry if less than all of them
    
    matrix(c(
      h2o.auc(m, TRUE, TRUE), pf@metrics$AUC,
      mcmsT[maix, 'value'], mcmsV[maix, 'value'], tms[ix, 'accuracy'],
      h2o.logloss(m, TRUE, TRUE), pf@metrics$logloss,
      h2o.mse(m, TRUE, TRUE), pf@metrics$MSE
    ), ncol = 4)
  }, simplify = "array")
  
  dimnames(res) <- list(
    c("train","valid","test"),
    c("AUC","Accuracy","logloss", "MSE"),
    labels
  )
  
  res
}
h2o.experiment(h2o.glm, list(family = "binomial"))


fTable1 <- kable(forec.table.h6.p1,digits=0 , format = "markdown", padding = 2, caption="Forecast results for candidate models")
fTable2 <- kable(forec.table.h6.p2,digits=0 , format = "markdown", padding = 2, caption="Forecast results for candidate models")
print(fTable1)
print(fTable2)
dl.fit <- h2o.experiment(h2o.deeplearning)
stTime <- Sys.time()

# Load Packages
pkg <- c("ggplot2", "knitr", "markdown", "glmnet", "caret", "gglasso", "corrplot","RColorBrewer","zoo", "factoextra", "mlbench", "rlang")

sapply(pkg, require, character.only = TRUE)

# Knitr Options
knitr::opts_chunk$set(progress = TRUE, fig.width=11, fig.height=6, echo = FALSE, message = FALSE, warning = FALSE, fig.align = "center", cache = TRUE,  cache.lazy = TRUE,  tidy=TRUE, tidy.opts=list(blank=TRUE, width.cutoff=65) )


# Setup logical directory structure
ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/"
DATA.DIR <- paste(ROOT.DIR,"data",sep="")
CHART.DIR <- paste(ROOT.DIR, "charts", sep="")
CODE.DIR <- paste(ROOT.DIR, "code", sep="")
# Import Data
data(Sonar)

# Investigate Sonar Data Set
# str(Sonar)

seed = 777
set.seed(seed)
# reomve rows with missing values
Sonar = Sonar[complete.cases(Sonar),]
Sonar = Sonar[sample(nrow(Sonar)),]

# Assign Predictors and Response
y = Sonar$Class
X = Sonar[,-61]

# Correlation Plot
# create correlation matrix
cor_mat=cor(X, use="complete.obs")
# plot cor matrix
corrplot(cor_mat, 
         order = "original", 
         method="square") 
# Apply Hierarchical Clustering

# Dissimilarity matrix
d = dist(t(X), method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 = hclust(d, method = "complete" )

# Apply Elbow rule
fviz_nbclust(X, FUN = hcut, method = "wss") # It seems 4 - 6 groups may be appropriate

# Plot dendrogram with 5 selected groups
plot(hc1, cex = 0.6, hang = -1)
rect.hclust(hc1, k = 5, border = 2:5) 

# We can see that 5 groups is a reasonable choice

# Create groups for Group LASSO
# Cut tree into 5 groups
sub_grp = cutree(hc1, k = 5)
grp=c(as.matrix(sub_grp))
# Define Groups
grp1 = X[,names(which(sub_grp == 1))]
grp2 = X[,names(which(sub_grp == 2))]
grp3 = X[,names(which(sub_grp == 3))]
grp4 = X[,names(which(sub_grp == 4))]
grp5 = X[,names(which(sub_grp == 5))]
# Create a 90/10 Train Test Split
# set.seed(777)


trainIndex <- createDataPartition(y, p = .9, 
                                  list = FALSE, 
                                  times = 1)
# Seperate Train and Test Sets
y_train = y[trainIndex]
y_test = y[-trainIndex]
X_train = X[trainIndex,]
X_test = X[-trainIndex,]
#######################################
## Multi-Layer Perceptron
#######################################
# Define Grid
layer1 = seq(10,30,10)
layer2 = seq(10,30,10)
layer3 = seq(10,30,10)
mlp_grid = expand.grid(.layer1 = layer1, 
                       .layer2 = layer2, 
                       .layer3 = layer3)

# Train Model
run.MLP = 0
if(run.MLP == 1){
  set.seed(1)
  start.time.mlp <- Sys.time()
  fit.mlp = caret::train(x = data.matrix(X_train),
                         y = y_train, 
                         method = "mlpML", 
                         #trControl = trnCtrl,
                         #act.fct = 'logistic',
                         tuneGrid = mlp_grid,
                         standardize = FALSE)
  
  end.time.mlp <- Sys.time()
  time.taken.mlp <- end.time.mlp - start.time.mlp 
  
  
  saveRDS(time.taken.mlp,
          file = paste(DATA.DIR, "time.taken.mlp.rds", sep="/"))
  save(fit.mlp,
       file = paste(DATA.DIR, "fit.mlp.Rdata", sep="/"))
  saveRDS(fit.mlp,
          file = paste(DATA.DIR, "fit.mlp.rds", sep="/") )
}


time.taken.mlp <- readRDS(file = paste(DATA.DIR, "time.taken.mlp.rds", sep="/"))
# fit.cv.ridge <- readRDS(file = paste(DATA.DIR, "fit.cv.ridge.rds", sep="/"))
load(file = paste(DATA.DIR, "fit.mlp.Rdata", sep="/"))


# Plot and Model Details
#plot(my.train)
best.params = fit.mlp$bestTune
my.mlp.model <- fit.mlp$finalModel
# Prediction on Test Set
pred.mlp = predict(fit.mlp, newdata = data.matrix(X_test))
mlp_matrix <- caret::confusionMatrix(pred.mlp, y_test)
#######################################
## LASSO
#######################################
# Define Grid
lambda.grid <- seq(0, 50)
alpha.grid <- seq(0, 1, length = 20)

srchGrd = expand.grid(.alpha = 1, .lambda = lambda.grid)

# Setup CV Function
trnCtrl = trainControl(
  method = "repeatedCV",
  number = 10,
  repeats = 5)

# Train Model
run.LASSO = 0
if(run.LASSO == 1){
  set.seed(seed)
  start.time.L <- Sys.time()
  
  fit.LASSO <- caret::train(x = data.matrix(X_train),
                            y = y_train,
                            method = "glmnet",
                            tuneGrid = srchGrd,
                            trControl = trnCtrl,
                            standardize = FALSE)
  
  
  end.time.L <- Sys.time()
  time.taken.L <- end.time.L - start.time.L


  saveRDS(time.taken.L,
          file = paste(DATA.DIR, "time.taken.L.rds", sep="/"))
  save(fit.LASSO,
       file = paste(DATA.DIR, "fit.LASSO.Rdata", sep="/"))
  saveRDS(fit.LASSO,
          file = paste(DATA.DIR, "fit.LASSO.rds", sep="/") )
}

time.taken.L <- readRDS(file = paste(DATA.DIR, "time.taken.L.rds", sep="/"))
# fit.LASSO <- readRDS(file = paste(DATA.DIR, "fit.LASSO.rds", sep="/"))
load(file = paste(DATA.DIR, "fit.LASSO.Rdata", sep="/"))


# Plot and Model Details
best.params = fit.LASSO$bestTune
my.LASSO.model <- fit.LASSO$finalModel
# Prediction on Test Set
pred.LASSO = predict(fit.LASSO, newdata = data.matrix(X_test))
LASSO_matrix <- caret::confusionMatrix(pred.LASSO, y_test)
#######################################
## Elastic Net (tuned alpha)
#######################################
# Define Grid
lambda.grid <- seq(0, 50)
alpha.grid <- seq(0, 1, length = 20)

srchGrd = expand.grid(.alpha = alpha.grid, .lambda = lambda.grid)

# Setup CV Function
trnCtrl = trainControl(
  method = "repeatedCV",
  number = 10,
  repeats = 5)

# Train Model
run.fit.glmN = 0
if(run.fit.glmN == 1){
  # set.seed(3)
  set.seed(seed)
  start.time.EN <- Sys.time()
  fit.glmN <- caret::train(x = data.matrix(X_train),
                           y = y_train,
                           method = "glmnet",
                           tuneGrid = srchGrd,
                           trControl = trnCtrl,
                           standardize = FALSE)
  
  end.time.EN <- Sys.time()
  time.taken.EN <- end.time.EN - start.time.EN

  saveRDS(time.taken.EN,
          file = paste(DATA.DIR, "time.taken.EN.rds", sep="/"))
  save(fit.glmN,
       file = paste(DATA.DIR, "fit.glmN.Rdata", sep="/"))
  saveRDS(fit.glmN,
          file = paste(DATA.DIR, "fit.glmN.rds", sep="/") )
  }



time.taken.EN <- readRDS(file = paste(DATA.DIR, "time.taken.EN.rds", sep="/"))
# fit.glmN <- readRDS(file = paste(DATA.DIR, "fit.glmN.rds", sep="/"))
load(file = paste(DATA.DIR, "fit.glmN.Rdata", sep="/"))


# Plot and Model Details
#plot(fit.glmN)
best.params = fit.glmN$bestTune
my.glmnet.model <- fit.glmN$finalModel
# Prediction on Test Set
pred.glmnet = predict(fit.glmN, newdata = data.matrix(X_test))
glmN_matrix = caret::confusionMatrix(pred.glmnet, y_test)

#######################################
## Group LASSO
#######################################
# Cross Validation (This can take a substantial amount of time - around 45 mins)
X_train = as.matrix(X_train)
y_train = ifelse(y_train == "M", 1 , -1) # Binary Factor needs to be numeric {-1,1} format

run.GLASSO = 0 
if(run.GLASSO == 1){
  # set.seed(4)
  set.seed(seed)
  start.time.GLASSO <- Sys.time()
  fit.GLASSO=cv.gglasso(x=X_train,
                        y=y_train,
                        group=grp, 
                        loss="logit",
                        pred.loss="L1", # Penalized Logistic Regression
                        nfolds=10) 
  
  end.time.GLASSO <- Sys.time()
  time.taken.GLASSO <- end.time.GLASSO - start.time.GLASSO
  
  saveRDS(time.taken.GLASSO,
          file = paste(DATA.DIR, "time.taken.GLASSO.rds", sep="/"))
  save(fit.GLASSO,
       file = paste(DATA.DIR, "fit.GLASSO.Rdata", sep="/"))
  saveRDS(fit.GLASSO,
          file = paste(DATA.DIR, "fit.GLASSO.rds", sep="/") )
  }

time.taken.GLASSO <- readRDS(file = paste(DATA.DIR, 
                                          "time.taken.GLASSO.rds", sep="/"))
# fit.cv.ridge <- readRDS(file = paste(DATA.DIR, "fit.cv.ridge.rds", sep="/"))
load(file = paste(DATA.DIR, "fit.GLASSO.Rdata", sep="/"))



# Prediction
pred.gglasso = predict(fit.GLASSO, newx = data.matrix(X_test), 
                       s = "lambda.min", type = "class")
pred.gglasso = ifelse(pred.gglasso == 1, "M", "R")
pred.gglasso = as.factor(pred.gglasso)
GLASSO_matrix = caret::confusionMatrix(pred.gglasso, y_test)

#######################################
# Group Ridge
#######################################
# Cross Validation (This can take several minutes)

run.RLASSO = 0
if(run.RLASSO == 1){
  # set.seed(5)
  set.seed(seed)
  start.time.RLASSO <- Sys.time()
  fit.cv.ridge = cv.gglasso(x=X_train, 
                            y=y_train, 
                            group=grp, 
                            loss="logit",
                            pred.loss="L2", # Penalized Logistic Regression
                            nfolds=10)
  
  end.time.RLASSO <- Sys.time()
  time.taken.RLASSO <- end.time.RLASSO - start.time.RLASSO
  
  
  saveRDS(time.taken.RLASSO, 
          file = paste(DATA.DIR, "time.taken.RLASSO.rds", sep="/"))
  save(fit.cv.ridge, 
       file = paste(DATA.DIR, "fit.cv.ridge.Rdata", sep="/"))
  saveRDS(fit.cv.ridge, 
          file = paste(DATA.DIR, "fit.cv.ridge.rds", sep="/") )
}


time.taken.RLASSO <- readRDS(file = paste(DATA.DIR, "time.taken.RLASSO.rds", sep="/"))
# fit.cv.ridge <- readRDS(file = paste(DATA.DIR, "fit.cv.ridge.rds", sep="/"))
load(file = paste(DATA.DIR, "fit.cv.ridge.Rdata", sep="/"))

# Best Lambda
lmbda=fit.cv.ridge$lambda.min

# Prediction
pred.gglasso.L2 = predict(fit.cv.ridge, newx = data.matrix(X_test), 
                       s = "lambda.min", type = "class")
pred.gglasso.L2 = ifelse(pred.gglasso.L2 == 1, "M", "R")
pred.gglasso.L2 = as.factor(pred.gglasso.L2)
RGroup_matrix = caret::confusionMatrix(pred.gglasso.L2, y_test)
# LASSO CV Plot
# plot(fit.LASSO)


svg(file = paste(CHART.DIR, "fit.LASSO.svg", sep="/"), width = 8, height =6, pointsize=12)
plot(fit.LASSO)
# mtext(adj=0, side=3 , 
      # text=expression(bold(paste("LASSO Cross Validation - ", (beta)," * SPY"))))
dev.off()
# Best Lambda Value - Group LASSO
GLASSO.lmbda=log(fit.GLASSO$lambda.min)
# Group LASSO Model Plots
plot(fit.GLASSO$gglasso.fit,
     xlim = c(-6,-3.75),
     ylim = c(-2,2.5))
abline(v = GLASSO.lmbda)
text(-5.8,2.2, bquote(lambda == .(round(GLASSO.lmbda,3))))
plot(coef(fit.GLASSO)[-1],
     ylab = "Coefficient Value",
     xlab = "Variable",
     main = "Coefficient: Group LASSO")
text(44,1.2, "Partial Group 1 Shrinkage", col = "Red")
abline(v=c(51,60), h=0, col = c("Black","Red","Red"))
# Elastic Net Plot
plot(fit.glmN)
# Confusion Matrices
mlp_matrix
LASSO_matrix
glmN_matrix
GLASSO_matrix
RGroup_matrix
stTime 

endTime <- Sys.time()
endTime

endTime   - stTime
```




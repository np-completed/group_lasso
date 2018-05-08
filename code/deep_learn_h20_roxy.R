#'  LIBRARIES for running h2o  
#' 

#+ importLibraries,echo=TRUE,results='hide',cache=FALSE
# Download packages that H2O depends on.
pkgs <- c("pROC", "RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
# Load libs
pkg.load <- c("AppliedPredictiveModeling", "bitops", "caret", "dplyr", "ggbiplot", "h2o", "mlbench", "pROC", "RCurl", "rjson", "ROCR", "statmod", "tools")
sapply(pkg.load, require, character.only = TRUE)



#' **Helper File** for DeepLearning Model
#' 

#+ helperFile,echo=TRUE
source("/media/disc/Megasync/R/regularization/group_lasso/code/helper_h2o.R")


#+ directory_shorcuts
ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/"
DATA.DIR <- paste(ROOT.DIR,"data",sep="")
CODE.DIR <- paste(ROOT.DIR, "code", sep="")


#' **MAKE RESULTS REPEATABLE**
#' 

#+ seed_number, echo=TRUE,cache=FALSE
seed = 0
set.seed(seed)


#' ## INITIALIZE H2O 
#' 
#' H2O is a Java application. It runs a web server and interacts with R
#' via REST-like web services. The command below starts the H2O. The nthreads
#' parameter value -1 tells H2O to run with as many threads as there are
#' CPU cores.
#' 
#' First connect to a local H2O instance from RStudio using all CPUs and 14 gigabytes of memory. 
#'  

#+ h20_config, echo=TRUE, cache=FALSE
h2o <- h2o.init(nthreads = -1, max_mem_size = "14g" )


#' LOAD SONAR DATA AND CREATE TRAINING AND TEST SETS  
#' 


#+ data_input_DL
# data(Sonar)
# write.csv(Sonar, file = paste(DATA.DIR, "sonar_dl.csv", sep="/"), row.names=FALSE )
Sonar <- read.csv(file = paste(DATA.DIR, "sonar_dl.csv", sep="/"))
#
pathToData <- paste0(normalizePath(DATA.DIR), "/sonar_dl.csv")


#' 1. Load data into an H2O data frame
#' 1. Optional: Convert the H2O data frame into an R data frame for later use.
#' 1. Create the feature vectors by dropping the classifications.
#' 

#+ sonarHex, echo=TRUE,cache=FALSE,results='hide'
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


#' Use H2O to create the traning and test data sets from the original data.
#' 
#' * 80% Train 
#' 
#' * 20% Test
#' 
#' Pass seed value generated previously to make the results repeatable.
#' 

#+ train_test_DL,echo=TRUE,cache=FALSE
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


#' ### TRELLIS PLOT   
#' 
#' Check data quality to ensure everything went good so far. 
#' Create a trellis plot using all 60 features. Notice that it is a useless mess.
#' 

#+ plt_trellis, echo=TRUE
transparentTheme(trans = 0.5)
featurePlot <- featurePlot(
  x = sonar.features,
  y = sonar.df$Class,
  plot = "pairs",
  pscales = FALSE,
  cex = 0.05,
  auto.key = list(columns = 2) )
featurePlot


#' ## PRINCIPAL COMPONENT ANALYSIS  
#' 
#' Generally, anything with more than 10 features is considered high dimensional data.
#' Text mining and genomic microarrays have thousands of dimensions and they are the
#' go-to examples of high dimensional data.
#' ---


#+ pca_fit_dl, echo=TRUE
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


#' Plot the first three principal components. Together, the first three
#' principle components accounts for almost 50% of the variance in the data!
#' That means they are very rich in information. After the PCA transformation,
#' The original data set contains 60 features. If we think of the first three principal
#' components as features, then 3 features can account for 50% of the information in the
#' original data set. That means the the first three principle components consume only
#' 3/60, or just 5% of the space as the original data, but contains almost half the signal.
#' The upshot is that PCA is a kind of lossy data compression! 
#' 
#' ### PCA Cumulative Variance
#' 
#' Look at cumulative proportion of the variance explained by the principle components:

#+ pca_variance, echo=TRUE
summary(sonar.princomp.7)
# PCA TRELLIS PLOT 
transparentTheme(trans = 0.3)
numberOfPrincipalComponentsToPlot = 3


#' Plotting the principle components shows us that the rocks and mines have a lot of
#' overlap. The more overlaps, the harder it will be for the machine learning algorithm
#' to distinguish the difference between rocks and mines. However, it is easy to see the
#' two classes do not overlap entirely; there is definitely some signal for the algorithm
#' to lock onto.

#+ plt_PCA_feature, echo=TRUE
featurePlot(
  x = sonar.princomp$x[, 1:numberOfPrincipalComponentsToPlot],
  y = sonar.df$Class,
  plot = "pairs",
  pscales = FALSE,
  auto.key = list(columns = 2)
)


#' ### PLOT FIRST AND SECOND PRINCIPLE COMPONENTS 
#' 
#' Use a special libary to plot the first principle component against the second
#' principle component. Draw a ellipses around the observations. 
#' 
#' * Observations inside the ellipse are less than 1 std dev away from mean.
#' 


#+ plt_PCA_loadings, echo=TRUE
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



#' # Trees {.tabset .tabset-fade}
#' 
#' ## Decision Tree
#' 
#' Let's see how a decision tree does. Having a baseline to compare accuracy is useful. 
#' 


#+ decision_tree, echo=TRUE, cache=FALSE
decisionTree = train(Class ~ .,  sonar.train.df, method="ctree")
# Visually inspect the tree.
plot(decisionTree$finalModel)


#' Evaluate the decision tree performance
#' 

#+ decision_tree_results, cache=FALSE, echo=TRUE
decisionTree.predictions = predict(decisionTree, sonar.test.df[, -classVariableIndex])
confusionMatrix(
  decisionTree.predictions,
  sonar.test.df$Class)


#' ## AUC results
#' 
#' The decision tree area-under-curve (AUC) results look good. Reported 84%  
#' 

#+ auc_results, fig.height=5, fig.width=5, echo=TRUE, cache=FALSE
decisionTree.predictions.probabilities=predict(decisionTree, sonar.test.df[, -classVariableIndex], type="prob")
decisionTree.roc=roc(sonar.test.df$Class, decisionTree.predictions.probabilities$M)
# Area under the curve: 0.8433
plot(decisionTree.roc)



#' ## Random Forest and GBM
#' 
#'   
#'  1. Gradient Boosted Model AUC=91%
#'  1. Random Forest
#'  
#' Ensemble learners (EL) are usually composed of decision trees.
#'  An EL combines multiple weak learners into a single strong learner.
#'  See bias-variance tradeoff for better understanding.  
#'  


#+ gbm_fit, echo=TRUE, cache=cacheData
model.gbm.exp <- h2o.experiment(h2o.gbm)
m.gbm0 <- h2o.gbm(1:60, 61, sonar.train,
                  nfolds = 0 ,model_id = "GBM_default")
m.gbm0

#+ randForest_fit, echo=TRUE, cache=cacheData
# Random forest. A good ensemble learner.
model.rf.exp <- h2o.experiment(h2o.randomForest)
m.rf0 <- h2o.randomForest(1:60, 61,
                              sonar.train, 
                              nfolds = 0, 
                              model_id = "RF_default")
summary(m.rf0)
h2o.performance(m.rf0, sonar.test)



#+ fit_compare, echo=TRUE
res <- compareModels(c(m.gbm0, m.rf0), sonar.test)
round(res[,"AUC",], 3)

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


compareModels <- function(models, test, labels=NULL){
  if(is.null(labels)){
    labels <- lapply(models, function(m) m@model_id)
  }
  
  res <- sapply(models, function(m){
    mcmsT <- m@model$training_metrics@metrics$max_criteria_and_metric_scores
    mcmsV <- m@model$validation_metrics@metrics$max_criteria_and_metric_scores
    maix <- which(mcmsT$metric=="max accuracy")
    th <- mean(mcmsT[maix, 'threshold'], mcmsV[maix,'threshold'])
    
    pf <- h2o.performance(m, test)
    tms <- pf@metrics$thresholds_and_metric_scores
    ix <- apply(outer(th, tms$threshold, "<="),1, sum)
    if(ix < 1)ix <- 1
    
    matrix(c(
      h2o.auc(m, TRUE, TRUE), pf@metrics$AUC,
      mcmsT[maix, 'value'], mcmsV[maix, 'value'], tms[ix, 'accuracy'],
      h2o.logloss(m, TRUE, TRUE), pf@metrics$logloss,
      h2o.mse(m, TRUE, TRUE), pf@metrics$MSE
    ), ncol = 4)
  }, simplify = "array")
  
  dimnames(res) <- list(
    c("train", "valid", "test"),
    c("AUC", "Accuracy", "logloss", "MSE"),
    labels
  )
  
  res
}


#' ## GLM
#' 
#'  Generalized linear model.
#'   
#' 1. GLM AUC=94% 
#'  


#+ glm_fit, echo=TRUE, cache=cacheData
h2o.experiment(h2o.glm, list(family = "binomial"))


#+ forecTa, results='asis' , eval=FALSE
fTable1 <- kable(forec.table.h6.p1,digits=0 , format = "markdown", padding = 2, caption="Forecast results for candidate models")
fTable2 <- kable(forec.table.h6.p2,digits=0 , format = "markdown", padding = 2, caption="Forecast results for candidate models")
print(fTable1)
print(fTable2)

#' ## Deep Learning Model Fit
#' 
#' Deep learning (DL) can be considered as a neural network that includes extra hidden layers. These hidden layers account for nonlinear relationships and latent variance. Raw input is automatically converted to good features. The increased accuracy
#' comes at a computationally expensive cost. Tuning the model parameters will have 
#' dramatic effects on the accuracy. An untuned DL model will generally perform poorly, with low accuracy on test set predictions.
#' 


#+ deep_learning_fit,echo=TRUE,cache=FALSE
dl.fit <- h2o.experiment(h2o.deeplearning)



#' ## Model Results
#' 
#' Comparing performance accuracy from models run in h2o. 
#' 

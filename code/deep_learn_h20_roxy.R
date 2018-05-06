#' ## LIBRARIES for running h2o  
#' 

#+ importLibraries, echo=TRUE, results='hide',cache=FALSE
pkg <- c("AppliedPredictiveModeling", "bitops", "caret", "dplyr", "ggbiplot", "h2o", "mlbench", "pROC", "RCurl", "rjson", "ROCR", "statmod", "tools")
sapply(pkg, require, character.only = TRUE)



#' ## Helper File for DeepLearning Model
#' 

#+ helperFile,echo=TRUE
source("/media/disc/Megasync/R/regularization/group_lasso/code/helper_h2o.R")


#+ directory_shorcuts
ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/code/"
DATA.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/data/" 

#' ## MAKE RESULTS REPEATABLE
#' 

#+ seed_number, echo=TRUE,cache=FALSE
seed = 0
set.seed(seed)


#' INITIALIZE H2O --------------------------------------------------------------  
#' 
#' H2O is a Java application. It runs a web server and interacts with R
#' via REST-like web services. The command below starts the H2O. The nthreads
#' parameter value -1 tells H2O to run with as many threads as there are
#' CPU cores.


#' ## h20 Config 
#' 

#+ h20_config, echo=TRUE, cache=FALSE
h2o <- h2o.init(nthreads = -1 )

#' LOAD SONAR DATA AND CREATE TRAINING AND TEST SETS ---------------------------
#' 

#+ data_input_DL
# data(Sonar)
pathToData <- paste0(normalizePath(DATA.DIR), "sonar_dl.csv")

#' Load data into an H2O data frame
#' 

#+ sonarHex, echo=TRUE,cache=FALSE
# sonar.hex = h2o.importFile("/media/disc/Megasync/R/regularization/group_lasso/data/sonar_dl.csv") 
sonar.hex <- h2o.importFile(path = pathToData, destination_frame = "dat")
# Convert the H2O data frame into an R data frame for later use.
sonar.df = as.data.frame(sonar.hex)
# sonar.df <- select(sonar.df, -C1)

#' Create the feature vectors by dropping the classifications
#' 

#+ featureVectors, echo=TRUE,cache=FALSE
classVariableIndex = 61
sonar.features = sonar.df[, -classVariableIndex]
# Rename the classification varible to something meaningful
names(sonar.df)[classVariableIndex] = "Class"

#' Use H2O to create the traning and test data sets from the original data.
#' Use 80% of the data for training and 20% for testing. Pass in the seed value
#' to make the results repeatable.
#' 

#+ train_test_DL,echo=TRUE,cache=FALSE
sonar.split = h2o.splitFrame(data = sonar.hex,
                             ratios = 0.8,
                             seed = seed)
# Create named varaibles for the traning and test data
sonar.train = sonar.split[[1]]
sonar.test = sonar.split[[2]]
# Create R data frames from the H2O data frames
sonar.train.df = as.data.frame(sonar.train)
names(sonar.train.df)[classVariableIndex] = "Class"
sonar.test.df = as.data.frame(sonar.test)
names(sonar.test.df)[classVariableIndex] = "Class"


#' ## TRELLIS PLOT ----------------------------------------------------------------
#' 
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

#' Extract all x variables
#' x.vars <- sonar.features %>% select(-C1)

#' PRINCIPAL COMPONENT ANALYSIS OF DATA SET ------------------------------------  
#' 
#' Generally, anything with more than 10 features is considered high dimensional data.
#' Text mining and genomic microarrays have thousands of dimensions and they are the
#' go-to examples of high dimensional data.
#' ---


#+ pca_fit_dl, echo=TRUE
# Use principal components to reduce the dimensionality of the features.
sonar.princomp = prcomp(sonar.features, #x.vars,  
                        center = TRUE,
                        scale = TRUE,
                        retx = TRUE)



#' Plot the first three principal components. Together, the first three
#' principle components accounts for almost 50% of the variance in the data!
#' That means they are very rich in information. After the PCA transformation,
#' The original data set contains 60 features. If we think of the first three principal
#' components as features, then 3 features can account for 50% of the information in the
#' original data set. That means the the first three principle components consume only
#' 3/60, or just 5% of the space as the original data, but contains almost half the signal.
#' The upshot is that PCA is a kind of lossy data compression!

#+ pca_variance, echo=TRUE
# Look at cumulative proporation of the variance explained by the principle components
# The first
summary(sonar.princomp)
# PCA TRELLIS PLOT ++++++++++++++++++++
transparentTheme(trans = 0.3)
numberOfPrincipalComponentsToPlot = 4


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


#' ### PLOT FIRST AND SECOND PRINCIPLE COMPONENTS ----------------------------------
#' 
#' Use a special libary to plot the first principle component against the second
#' principle component. Draw a ellipses around the observations. Observations
#' inside the ellipse are less than 1 std dev away from mean.
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


#' ## MACHINE LEARNING EXPERIMENTS ------------------------------------------------
#' 

#' ### Decision Tree  
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
#' The area-under-curve (AUC) results look good.
#' 

#+ auc_results, echo=TRUE, cache=FALSE
decisionTree.predictions.probabilities=predict(decisionTree, sonar.test.df[, -classVariableIndex], type="prob")
decisionTree.roc=roc(sonar.test.df$Class, decisionTree.predictions.probabilities$M)
# Area under the curve: 0.8433
plot(decisionTree.roc)

#' Now we pull out the big guns and feed the problem to H2O's algorithms.
#' 
#' Review ROC plots. Explain it in terms of classifying rocks vs mines.
#' The more tolerant you are of "slop" (false positives), the more you can be sure
#' you aren't missing any true positives. The more willing you are to
#' accept the computer misclassifying, the less
#' likely you are to miss-classify a mine.
#'
#' PS: ROC curve technical background requires some understanding. I've been meaning to read
#' these articles about it:
#' http://blogs.sas.com/content/iml/2011/07/29/computing-an-roc-curve-from-basic-principles.html
#' http://www.dataschool.io/roc-curves-and-auc-explained/
#' 

#' # Gradient boosted machines.
#' 
#' Ensemble learners (EL) are usually composed of decision trees.
#'  An EL combines multiple weak learners into a single storng learner. See bias-variance 
#'  tradeoff for better understanding.  
#'  

#+ ensemble_fit, echo=TRUE, cache=FALSE
h2o.experiment(h2o.gbm)
# Generalized linear model.
h2o.experiment(h2o.glm, list(family = "binomial"))
# Random forest. A good ensemble learner.
h2o.experiment(h2o.randomForest)

#' ## Deep Learning Model Fit  -----------------------------
#' 
#' Deep learning can be considered as a neural network that includes extra layers, e.g. 
#' hidden layers. Raw input is automatically converted to good features. The accuracy
#' comes at a computationally expensive cost. Tuning the model parameters will have 
#' dramatic effects on the accuracy.
#' 


#+ deep_learning_fit,echo=TRUE,cache=FALSE
h2o.experiment(h2o.deeplearning)

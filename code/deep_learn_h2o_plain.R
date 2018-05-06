## SETUP -----------------------------------------------------------------------

# Install these packages if necessary
# library(devtools)
# install_github("vqv/ggbiplot")


# Next, we download packages that H2O depends on.
pkg.install <- c("bitops", "RCurl","jsonlite", "party", "rjson", "statmod", "tools")
for (pkg in pkg.install) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
  }

# LIBRARIES
pkg <- c("AppliedPredictiveModeling", "bitops", "caret", "dplyr", "ggbiplot", "h2o", "mlbench", "pROC", "RCurl", "rjson", "ROCR", "statmod", "tools")
sapply(pkg, require, character.only = TRUE)



## Helper File for DeepLearning
source("/media/disc/Megasync/R/regularization/group_lasso/code/helper_h2o.R")


ROOT.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/code/"
DATA.DIR <- "/media/disc/Megasync/R/regularization/group_lasso/data/" 

# MAKE RESULTS REPEATABLE
seed = 0
set.seed(seed)


# INITIALIZE H2O --------------------------------------------------------------
# H2O is a Java application. It runs a web server and interacts with R
# via REST-like web services. The command below starts the H2O. The nthreads
# parameter value -1 tells H2O to run with as many threads as there are
# CPU cores.
h2o <- h2o.init(nthreads = -1 )

# LOAD SONAR DATA AND CREATE TRAINING AND TEST SETS ---------------------------
# data(Sonar)
pathToData <- paste0(normalizePath(DATA.DIR), "sonar_dl.csv")
# write.table(x = Sonar, file = pathToData, row.names = F, col.names = T)

# Load data into an H2O data frame
# sonar.hex = h2o.importFile("/media/disc/Megasync/R/regularization/group_lasso/data/sonar.csv")
# sonar.hex = h2o.importFile("/media/disc/Megasync/R/regularization/group_lasso/data/sonar_dl.csv")
sonar.hex <- h2o.importFile(path = pathToData, destination_frame = "dat")

# Convert the H2O data frame into an R data frame for later use.
sonar.df = as.data.frame(sonar.hex)
# sonar.df <- select(sonar.df, -C1)

# Create the feature vectors by dropping the classifications
classVariableIndex = 61
sonar.features = sonar.df[, -classVariableIndex]

# Rename the classification varible to something meaningful
names(sonar.df)[classVariableIndex] = "Class"

# Use H2O to create the traning and test data sets from the original data.
# Use 80% of the data for training and 20% for testing. Pass in the seed value
# to make the results repeatable.
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

# TRELLIS PLOT ----------------------------------------------------------------
# Create a trellis plot using all 60 features. Notice that it is a useless mess.o
transparentTheme(trans = 0.5)
featurePlot <- featurePlot(
  x = sonar.features,
  y = sonar.df$Class,
  plot = "pairs",
  pscales = FALSE,
  cex = 0.05,
  auto.key = list(columns = 2) )

#' r 
## Extract all x variables
# x.vars <- sonar.features %>% select(-C1)


# PRINCIPAL COMPONENT ANALYSIS OF DATA SET ------------------------------------
# Generally, anything with more than 10 features is considered high dimensional data.
# Text mining and genomic microarrays have thousands of dimensions and they are the
# go-to examples of high dimensional data.
# ---
# Use principal components to reduce the dimensionality of the features.
sonar.princomp = prcomp(sonar.features, #x.vars,  
                        center = TRUE,
                        scale = TRUE,
                        retx = TRUE)

# Look at cumulative proporation of the variance explained by the principle components
# The first
summary(sonar.princomp)

# PCA TRELLIS PLOT ------------------------------------------------------------
transparentTheme(trans = 0.3)

# Plot the first three principal components. Together, the first three
# principle components accounts for almost 50% of the variance in the data!
# That means they are very rich in information. After the PCA transformation,
# The original data set contains 60 features. If we think of the first three principal
# components as features, then 3 features can account for 50% of the information in the
# original data set. That means the the first three principle components consume only
# 3/60, or just 5% of the space as the original data, but contains almost half the signal.
# The upshot is that PCA is a kind of lossy data compression!
numberOfPrincipalComponentsToPlot = 4


# Plotting the principle components shows us that the rocks and mines have a lot of
# overlap. The more overlaps, the harder it will be for the machine learning algorithm
# to distinguish the difference between rocks and mines. However, it is easy to see the
# two classes do not overlap entirely; there is definitely some signal for the algorithm
# to lock onto.
featurePlot(
  x = sonar.princomp$x[, 1:numberOfPrincipalComponentsToPlot],
  y = sonar.df$Class,
  plot = "pairs",
  pscales = FALSE,
  auto.key = list(columns = 2)
  )

# PLOT FIRST AND SECOND PRINCIPLE COMPONENTS ----------------------------------
# Use a special libary to plot the first principle component against the second
# principle component. Draw a ellipses around the observations. Observations
# inside the ellipse are less than 1 std dev away from mean.
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


# MACHINE LEARNING EXPERIMENTS ------------------------------------------------

# Let's see how a good old-fashioned decision tree does.
decisionTree = train(Class ~ .,  sonar.train.df, method="ctree")

# Visually inspect the tree.
plot(decisionTree$finalModel)

# Evaluate the decision tree's performance
decisionTree.predictions = predict(decisionTree, sonar.test.df[, -classVariableIndex])
confusionMatrix(
  decisionTree.predictions,
  sonar.test.df$Class)

# How good is our aread-under-curve (AUC)
decisionTree.predictions.probabilities=predict(decisionTree, sonar.test.df[, -classVariableIndex], type="prob")
decisionTree.roc=roc(sonar.test.df$Class, decisionTree.predictions.probabilities$M)
#Area under the curve: 0.8433
plot(decisionTree.roc)

# Now we pull out the big guns and feed the problem to H2O's algorithms.
# Review ROC plots. Explain it in terms of classifying images that contain dogs.
# The more tolerant you are of "slop" (false positives), the more you can be sure
# you aren't missing any true positives; i.e. dogs. The more willing you are to
# accept the computer misclassifying cats, monkeys and birds as dogs, the less
# likely you are to miss an images that actually contain a dog.
#
# PS: I don't know an ROC curve is acutally generated. I've been meaning to read
# these articles about it:
# http://blogs.sas.com/content/iml/2011/07/29/computing-an-roc-curve-from-basic-principles.html
# http://www.dataschool.io/roc-curves-and-auc-explained/

# Gradient boosted machines.
# Ensemble learner usually composed of decision trees.
# Ensemble learniner combines multiple weak learners into a single storng learner.  (Wikipedia)
h2o.experiment(h2o.gbm)

# Generalized linear model.
h2o.experiment(h2o.glm, list(family = "binomial"))

# Random forest. A good ensemble learner.
h2o.experiment(h2o.randomForest)

# Deep learning. I don't understand it well enough to explain how it works,
# but I like to think of it as a neural network that includes extra layers
# that take raw data and automatically discovery good features.
# It produces really good results. It is known to be computationally expensive.
h2o.experiment(h2o.deeplearning)




##### END OF PRESENTATION #####
#
# PCA EXPERIMENTS -------------------------------------------------------------
# Perform principle componennt analysis. In general, PCA does not improve the
# results delivered by H2O.

# Save the first seven principle components.
class.ind = 7
feature.ind = 1:(class.ind -1)

# Normalize the data before performing the SVD
pca = h2o.prcomp(
  sonar.hex[,1:60],
  k=class.ind,
  transform = "NORMALIZE",
  seed = seed
)
summary(pca)

# Create test and training results from the principle components
sonar.pca = h2o.predict(pca, sonar.hex)
sonar.pca.split = h2o.splitFrame(
  data = h2o.cbind(sonar.pca, sonar.hex[,61]),
  ratios = 0.8,
  seed=seed)
sonar.pca.train = sonar.pca.split[[1]]
sonar.pca.test = sonar.pca.split[[2]]

# Try GLM on the rotated test set.
# It performs awesome with the first seven principle components.
# Performance # gets worse with more or fewer components.
# max min_per_class_accuracy   0.93%
# max mean_per_class_accuracy  0.94%
# fit = h2o.glm(
#   x =feature.ind,
#   y = class.ind,
#   training_frame = sonar.pca.train,
#   family = "binomial")
# predictions = h2o.predict(fit, sonar.pca.test[,feature.ind])
# perf = h2o.performance(fit, sonar.pca.test)
# plot(perf)
# perf

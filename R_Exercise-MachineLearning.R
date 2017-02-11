# R_Exercise-MachineLearning.R
#
# Purpose:
#
# Version: 1.0
#
# Date:    2017  02  10
# Author:  Boris Steipe (boris.steipe@utoronto.ca)
#
# V 1.0    First code
#
# TODO:
#    https://rstudio.github.io/tensorflow/
#    https://www.r-bloggers.com/what-are-the-best-machine-learning-packages-in-r/
#    https://cran.r-project.org/web/views/MachineLearning.html
#
#
# == HOW TO WORK WITH THIS FILE ================================================
#
#  Go through this script line by line to read and understand the
#  code. Execute code by typing <cmd><enter>. When nothing is
#  selected, that will execute the current line and move the cursor to
#  the next line. You can also select more than one line, e.g. to
#  execute a block of code, or less than one line, e.g. to execute
#  only the core of a nested expression.
#
#  Edit code, as required, experiment with options, or just play.
#  Especially play.
#
#  DO NOT simply source() this whole file!
#
#  If there are portions you don't understand, use R's help system,
#  Google for an answer, or ask me. Don't continue if you don't
#  understand what's going on. That's not how it works ...
#
# ==============================================================================

# ==============================================================================
#        INTRODUCTION
# ==============================================================================

# This is a minimal example of using machine learning in R. We work with a well
# behaved dataset to set up a nicely defined classification problem. This will
# only serve to get you started in this huge field - but the things we practice
# here are not entirely trivial and I hope will give you ideas how to move on to
# tasks that are more interesting.

# The algorithms in the field are now quite powerful, mature and easy to use.
# However YOU have to contribute two elements, which are not trivial, and
# without which the approach will fail.

# YOU need to define good features.
# YOU need to define a good set of examples. Positive/negative - or in your
#     desired categories.


# ==============================================================================
#      PART ONE: A SIMPLE EXAMPLE WITH CRABS DATA USING THE caret PACKAGE.
# ==============================================================================

# The crabs dataset is one of the standard multivariate example datasets that
# ship with R - crabs, collected off Freemantle in West Australia are to be
# classified based on some morphometric measurements.

library(MASS)
data(crabs)

head(crabs)

# Lets make a combined factor column for species/sex as class labels and put
# that as the last column:
crabs$spsx <- as.factor(paste(crabs[, 1], crabs[, 2],sep="."))

str(crabs)
# Two species: blue and orange
# Two sexes:   female and male
#
# Rows   1: 50  blue males
# Rows  51:100  blue females
# Rows 101:150  orange males
# Rows 151:200  orange females
#
# FL frontal lobe size (mm)
# RW rear width (mm)
# CL carapace length (mm)
# CW carapace width (mm)
# BD body depth (mm)

# ==== EXPLORING THE DATA ======================================================

set.seed(112358)
N <- nrow(crabs)
rnd <- sample(1:N, N)  # That's just so we don't plot the points in order ...
plot(crabs[rnd, 4:8], pch=19, cex = 0.5, col=colorCrabs(crabs$spsx[rnd]))

# You get a sense that the data is separable - but in any individual dimension
# it's hard to do so. The solution in this case is to use Principal Components
# Analysis (PCA), to remove overall size as a confounding factor ...

pcaCrabs <- prcomp(crabs[, 4:8])
plot(pcaCrabs$x[rnd, 2],
     pcaCrabs$x[rnd, 3],
     pch=19, col=colorCrabs(crabs$spsx[rnd]))


# ... but that's not the topic of this exercise: it just goes to show that the
# data is indeed separable in principle, given a proper combination of the
# individual dimensions that have been measured. Machine learning automates the
# process of finding the discriminating features, and the thresholds for
# separation.


# ==== MACHINE LEARNING ========================================================


# To prepare our table for Machine learning, we drop the first three columns:
crabs <- crabs[,-(1:3)]
head(crabs)

# It is common for machine learning datasets to have the target categories
# (class labels) in the last column. Commonly, these are assumed to be factors.

# Can machine learning distinguish the crabs?
#
# We will use the caret package, which includes functions for cross-validation
# and tools for fitting. We load it with the (non-default) option of also
# loading "suggested" packages, which loads many, many packages that are useful
# for statistical learning and analysis.

if (! require(caret, quietly = TRUE)) {
    install.packages("caret", dependencies = c("Depends", "Suggests"))
    library(caret)
}

# Patience ...

# We will randomly remove 20% of each crabs category into a separate
# "validation" dataset. Machine learning operates on training- and test-data to
# optimize its parameters - but after we have built our models, we would still
# like to validate whether our prediction also works on completely "unknown"
# data.

set.seed(112358)
sel <- c(sample(  1:50,  10),
         sample( 51:100, 10),
         sample(101:150, 10),
         sample(151:200, 10))
crabsVal <- crabs[sel, ]
crabs <- crabs[-sel, ]
str(crabs)


# Define control parameters:
# 10-fold cross validation
myControl <- trainControl(method="cv", number=10)

# Accuracy: this is our target metric - correctly predicted instances vs. total
# instances in the test set, in %.
myMetric <- "Accuracy"


# Try a number of "typical" Machine Learning algorithms

# === linear algorithms
#     lda (linear discriminant analysis)
set.seed(112358)
fit.lda <- train(spsx~., data=crabs, method="lda",
                 metric=myMetric, trControl=myControl)

# We produce a number of fit. objects, and compare them all at the end:

# === nonlinear algorithms
# CART (Classification And Regression Trees)
set.seed(112358)
fit.cart <- train(spsx~., data=crabs, method="rpart",
                  metric=myMetric, trControl=myControl)

# kNN (k-Nearest Neighbours)
set.seed(112358)
fit.knn <- train(spsx~., data=crabs, method="knn",
                 metric=myMetric, trControl=myControl)

# === other algorithms
# SVM (Support Vector Machine)
set.seed(112358)
fit.svm <- train(spsx~., data=crabs, method="svmRadial",
                 metric=myMetric, trControl=myControl)

# Random Forest (often the "general purpose" method of first choice in ML)
set.seed(112358)
fit.rf <- train(spsx~., data=crabs, method="rf",
                metric=myMetric, trControl=myControl)


# Neural Network (magic)
set.seed(112358)
fit.nnet <- train(spsx~., data=crabs, method="nnet",
                metric=myMetric, trControl=myControl)


# == Evaluate
# summarize accuracy of models
myMLresults <- resamples(list(lda =  fit.lda,
                              cart = fit.cart,
                              knn =  fit.knn,
                              svm =  fit.svm,
                              rf =   fit.rf,
                              nnet = fit.nnet))
summary(myMLresults)

# The kappa statistic compares observed accuracy with expected accuracy, and
# thus takes into account that random chance may also give correct
# classifications. For a gentle, plain-english explanation see:
# http://stats.stackexchange.com/questions/82162/kappa-statistic-in-plain-english


dotplot(myMLresults)

# Linear discriminant analysis performed the best, both regarding accuracy and
# kappa statistic, with nnet a close second. Which method performs the best is
# obviously highly dependent on the data - and all of the methods allow
# optimizations ... this is a pretty big topic overall.

# How well did we do?
print(fit.lda)

# How can we use the classifier for predictions on unknown data? Remember that
# we had "unknown" data in our validation dataset. The functions called by caret
# are set up in similar ways as lm(), nls() or other modeling functions -
# specifically, they have a predict() method that allows to make predictions on
# new data with the larned parameters. Since we know the correct category labels
# in our validation set, we can easily check how often our prediction was right,
# or wrong: first we make predictions ...

myPredLDA <- predict(fit.lda, crabsVal)

# ... and then we analyze them in a "confusion matrix": predictions vs. known
# class labels.
confusionMatrix(myPredLDA, crabsVal$spsx)

# Not so bad - lda got all the orange crabs correct, and two of the blue ones
# wrong.

myPredNnet <- predict(fit.nnet, crabsVal)
confusionMatrix(myPredNnet, crabsVal$spsx)
# ... and nnet (with default parameters!) had only one error more.


# ==============================================================================
#      PART TWO: "INDUSTRY STRENGTH" ML WTIH h2o
# ==============================================================================

# h2o is a large, open platform for data science written in Java. After
# installing the package, an instance of h2o will run as a server for analysis
# and allow the R h2o package functions to interact with it. Installation from
# CRAN should be straightforward - even though the CRAN package has no actual
# h2o code, the required java "jar" file will be downloaded when the h2o.init()
# function is called for the first time.
#

if (! require(h2o, quietly = TRUE)) {
    install.packages("h2o")
    library(h2o)
}

H2O <- h2o.init()

# Prepare data again, to be sure ...
data(crabs)
crabs$spsx <- as.factor(paste(crabs[, 1], crabs[, 2],sep="."))
crabs <- crabs[,-(1:3)]
set.seed(112358)
sel <- c(sample(  1:50,  10),
         sample( 51:100, 10),
         sample(101:150, 10),
         sample(151:200, 10))
crabsVal <- crabs[sel, ]
crabs <- crabs[-sel, ]

# Prepare our dataset for h2o in h2o's .hex (hexadecimal) format:
crabs.hex <- as.h2o(crabs)
str(crabs.hex)

# Let's run a "Deep Neural Network" model (cf.
# https://en.wikipedia.org/wiki/Deep_learning for the concepts and vocabulary,
# also see http://docs.h2o.ai/h2o/latest-stable/h2o-docs/glossary.html for an
# h2o glossary) out of the box with all-default parameters:
( fit.h2o <- h2o.deeplearning(x = 1:5,
                              y = 6,
                              training_frame = crabs.hex,
                              seed = 112358) )

( myH2OPred <- h2o.predict(fit.h2o, as.h2o(crabsVal)) )
h2o.confusionMatrix(fit.h2o, as.h2o(crabsVal))

# This result is not very impressive - 20/40 errors! h2o separates orange from
# blue very well but does a terrible job at distinguishing male from female. cf.
# our lda result:
confusionMatrix(myPredictions, crabsVal$spsx)
# ... with only 2/40 errors.

# Try to improve these results by tuning the parameters: ten-fold
# cross-validation (default is none), four hidden layers of smaller size
# (default is c(200, 200)), set activation function to tanh (sigmoidal) (default
# is "Rectifier"), and use LOTS of iterations (default is only 10):
fit.h2o.2 <- h2o.deeplearning(
    x = 1:5,
    y = 6,
    training_frame = crabs.hex,
    hidden = c(8, 8, 8, 8),
    activation = "Tanh",
    seed = 112358,
    epochs = 10000,
    nfolds = 10,
    fold_assignment = "AUTO"
)
h2o.confusionMatrix(fit.h2o.2, as.h2o(crabsVal))

# Impressively, the result is now much, much better (only 1/40 errors - i.e.
# twice as good as lda). It's worthwhile to play around with the parameters aned
# see how they influence processing time and accuracy. You will find that not
# always will more layers, more nodes, more iterations lead to better results.
# You can also get a sense that you can burn A LOT of processing power building
# these models - but the results can also be very, very good. And that's the
# whole point, after all.

# Before you leave, don't forget to shut down the h2o server instance again, or
# it will keep on running in the background!
h2o.shutdown(prompt=FALSE)


# ==============================================================================
#      PART THREE: FEATURES
# ==============================================================================

# As noted in the introduction, the crabs example contains well behaved data.
# For much biological data the challenge is that features are categorical, and
# have huge numbers of states. For example genes, EC classifications, pathway
# membership, or GO terms. Here are some suggestions to address this:

# If the number of features is large, but their dimensions are real vealued, you
# can possibly reduce them with PCA. h2o can handle very large networks, but you
# may want to trade off network size for iterations (epochs).

# Categorical features can be turned into real values by "dummy coding". Instead
# of (male/female), make two features: isMale and isFemale and value them 0 or
# 1. (h2o actually does this automatically.)

# I would turn large categorical data into features by defining a number of "reference categories" - a set of genes, or well distinguished GO terms etc. and then calculating the similarity of my data point of interest to the references. For example, to turn the GO terms associated with TP53 into fatures, calculate the GO semantic similarities for each of the three ontologies to some, say, 50 different genes taken from very different aspects of cellular function, process and component.

# Levels can be combined: e.g. instead of full EC numbers use only the first two digits. Or, for levels of widely different numbers of membership, you can try combining by frequency: group the all the rare ones together, then the intermediate ones, then use the frequent ones as they are.

# If your data is binned into ranges, you can use the mean or median instead.



# ==== FURTHER READING =========================================================

# https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/deeplearning
# http://stats.stackexchange.com/questions/95212/improve-classification-with-many-categorical-variables
#



# ==============================================================================
#      BEYOND ...: PROJECTS
# ==============================================================================

# Try to distinguish Mbp1 target genes from Swi4 target genes - is that
# possible?






# [END]

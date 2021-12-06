# Machine Learning List
"""
Deep Learning - Deep learning is a type of machine learning and artificial intelligence (AI) that imitates the way humans gain certain types of knowledge. ... While traditional machine learning algorithms are linear, deep learning algorithms are stacked in a hierarchy of increasing complexity and abstraction.

  Deep Boltzmann Machine (DBM),
  Deep Belief Networks(DBN),
  Convolutional Neural Network (CNN),
  Stacked Auto-Encoders,

Neural Networks - A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes. Thus a neural network is either a biological neural network, made up of biological neurons, or an artificial neural network, for solving artificial intelligence problems.

  Radian Basis Function Network (RBFN),
  Perceptron,
  Back-Propagation,
  Hopfield Network,

Ensemble - Ensemble learning is the process by which multiple models, such as classifiers or experts, are strategically generated and combined to solve a particular computational intelligence problem. Ensemble learning is primarily used to improve the (classification, prediction, function approximation, etc.)

  Random Forest,
  Gradient Boosting Machines (GBM),
  Boosting,
  Bootstrapped Aggregation (Bagging),
  AdaBoost,
  Stacked Generalization (Blending),
  Gradient Boosted Regression Trees (GBRT),

Regularization - Image result for Regularization learning. This is a form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero. In other words, this technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.

  Ridge Regression,
  Least Absolute Shrinkage and Selection,
  Elastic Net,
  Least Angle Regression (LARS),

Rule System - RULE MODELS ARE the second major type of logical machine learning models. ... Particularly in supervised learning, a rule model is more than just a set of rules: the specification of how the rules are to be combined to form predictions is a crucial part of the model.

  Cubist,
  One Rule (OneR),
  Zero Rule (ZeroR),
  Repeated Incremental Pruning to Produce,
  Error Reduction (Ripper),

Regression - Regression is a supervised machine learning technique which is used to predict continuous values. The ultimate goal of the regression algorithm is to plot a best-fit line or a curve between the data. The three main metrics that are used for evaluating the trained regression model are variance, bias and error.Sep 4, 2019

  Linear Regression,
  Ordinary Least Squares Regression (OLSR),
  Stepwise Regression,
  Multivariate Adaptive Regression Splines (Mars),
  Locally Estimated Scatterplot SMoothing (LOESS),
  Logistic Regression,
  
Clustering - Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.

  k-Means,
  k-Medians,
  Expectation Maximization,
  Hierarchical Clustering,

Instance Based - Instance-based learning refers to a family of techniques for classification and regression, which produce a class label/predication based on the similarity of the query to its nearest neighbor(s) in the training set.

  k-Nearest Neighbour (kNN),
  Learning Vector Quantization (LVQ),
  Self-Organizing Map (SOM),
  Locally Weighed Learning (LWL),

Dimensionality Reduction - Dimensionality reduction, or dimension reduction, is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data, ideally close to its intrinsic dimension.

  Principal Component Analysis (PCA),
  Partial Least Squares Regression (PLSR),
  Sammon Mapping,
  Multidimensional Scaling (MDS),
  Projection Pursuit,
  Principal Component Regression (PCR),
  Partial Least Squares Discriminant Analysis,
  Mixture Discriminant Analysis(MDA),
  Quadratic Discriminant Analysis (QDA),
  Flexible Discriminant Analysis (FDA),
  Linear Discriminant Analysis (LDA),

Decision Tree - The tree can be explained by two entities, namely decision nodes and leaves. The leaves are the decisions or the final outcomes. And the decision nodes are where the data is split. An example of a decision tree can be explained using above binary tree.  Classification and Regression Tree (CART)

  Iterative Dichotomiser 3 (ID3),
  C 4.5,
  C 5.0,
  Chi-squared Automatic Interaction Detection (CHAID),
  Decision Stump,
  Conditional Decision Trees,
  M5,
  
Bayesian - The Bayesian framework for machine learning states that you start out by enumerating all reasonable models of the data and assigning your prior belief P(M) to each of these models. Then, upon observing the data D, you evaluate how probable the data was under each of these models to compute P(D|M)

  Naive Bayes,
  Averaged One-Dependence Estimators(AODE),
  Bayesian Belief Network (BBN),
  Gaussian Naive Bayes,
  Multinomial Naive Bayes,
  Bayesian Network (BN)
"""

# Name 1.Description 2.Adv 3.DisAdv
"""
Linear Regression
1. The best fit line through all data points
2. Easy to understand. You can clearly see what the biggest drivers of the model are.
3. Sometimes too simple to capture complex relationships between variables. Tendency for the model to overfit.

Logistic Regression
1. The adoption for linear regression to problems of classification
2. Easy to understand
3. Sometimes too simple to capture complex relationships between variables. Tendency for the model to overfit.

Decision Tree
1. A graph that uses branching method to match all possible outcomes of a decision
2. Easy to understand and implement
3. Not often used for prediction bc it's often to simple and not powerful enough for complex data

Random Forest
1. Takes the average of many decision trees. Each tree is weaker than the full decision tree, but combining them we get better overall performance.
2. Fast to train, and generally good quality results.
3. Can be slow to output predictions relative to other algorithms. Not easy to understand predictions

Gradient Boosting
1.Uses even weaker decision trees that increasingly focused on hard examples.
2. High Performing
3. A small change in the future set or training set can create radical changes in the model. Not easy to understand predictions.

Neural Networks
1.Mimics the behaviour of the brain. NNs are interconnected neurons that pass messages to each other. Deep learning uses seperate layers of NNs to put one after the other.
2. Can handle extremely complex tasks. No other algorithms comes close in image recognition
3.very very slow to train, bc they have so many layers. Requier a lot of power. Almost impossible to understand predictions.

"""

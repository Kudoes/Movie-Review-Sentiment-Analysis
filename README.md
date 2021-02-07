# 1 Introduction

This project focuses on utilizing thousands of labelled movie reviews to train a binary classification model
that will predict the sentiment of a new movie review based purely on the words in the review.

To accomplish this task, the following files are included in this repository:

1. `alldata.tsv`: This file contains the full dataset of reviews and their associated labels used to train and test the model.
2. `create_vocab.R`: This is the script used to generate a vocabulary from the training data. The terms within this vocabulary are used to train the logistic regression classifier.
3. `myvocab.txt`: This is the list of selected terms generated from the `create_vocab.R` script.
4. `mymain.R`: This file contains the code to train the model and evaluate the results (using AUC) on the test set.
5. `eval_code.R`: This is the script used to generate AUC results for each of the five splits.

## 1.1 Dataset

The dataset used in this project is similar to one used in the Kaggle competition “Bag of Words Meets
Bags of Popcorn”. The dataset consists of 50,000 total observations. Each observation consists of an id, score
(rating out of 10), a review (text-based review of the movie written by the reviewer), and a sentiment score
which is either 0 for negative, or 1 for positive. This dataset was used to generate five different training/test
splits. In each of the five splits, the data was split 50/50 so that each training set had 25,000 reviews, and
each test set had 25,000 reviews.

## 1.2 Goal

The goal of this project was to train a binary classification model that can predict the sentiment of a
movie review based purely on the words in the review itself. To evaluate the model performance, we aim to
maximize the AUC (of the ROC curve). To achieve this goal, we train a model using the training set (one of five at a time), 
and then evaluate the model performance on the test set using AUC as the critera. Our models would be
considered successful if (for each split) the AUC was equal to or greater than 0.96. Furthermore, we would
need to achieve this performance with a vocabulary size of less than 1000 words (or n-grams). In other words,
the feature set must be of size less than 1000, as the only features in this model are the words (or n-grams).

## 1.3 Approach

First, we need to generate a vocabulary size of less than 1000 words. Then, we create a document-term
matrix which consists of n rows (n = # of observations) and p columns (p = # of terms in the vocabulary).
For each observation, the column would contain the number of times the corresponding term appears in the
review. This matrix is used to fit the logistic regression model and then to make predictions on the test
data. 

# 2 Method

As previously mentioned, the first step of the project was to generate a vocabulary of less than 1000 terms.
The output of the vocabulary generation process is a text file of 684 lines, where each line corresponds to a selected term, where a term could be an n-gram up
to 4 words in length.

## 2.1 Preprocessing

Before fitting any models, the training data and test data have to be preprocessed (separately). These
steps included removing any HTML tags and invalid characters from the reviews themselves.

## 2.2 Vocabulary Generation

To generate the customized vocabulary, the following procedure was followed:

1. Load the “entire” dataset from `alldata.tsv`.
2. Remove special characters from all reviews.
3. Create the full vocabulary of all terms (1-to-4-grams) in the training data.
4. Prune uncommon words from this vocabulary.
5. Create a Document-Term Matrix of each training observation against the full vocabulary.
6. Use the two-sample t-test to identify the top 2000 positive/negative words from the full vocabulary.
7. Create a new vocabulary out of these new terms and create another DTM matrix.
8. Fit a LASSO regression model to this new DTM and select only the top 684 most important terms.
9. Save the resulting terms in a myvocab.txt file for the rest of the program.

After this process, we have a vocabulary size of 684 terms. Essentially, a two-sample t-test was used to identify the words in the vocabulary that were most strongly correlated with a positive/negative review. Next, a LASSO model was fit to those 2000 terms and L1 regularization was done to select only the top 684 most impactful terms from among those 2000.

## 2.3 Model Fitting and Tuning

Since we aim to make binary classifications of positive or negative reviews, we use logistic regression
and we fit the document-term matrix to the sentiment labels of each review. Logistic regression was ideal
as it is fast to train, and works quite well for this kind of task. To fit a logistic regression model, the
`cv.glmnet` function from `glmnet` was utilized. We use a penalty of `alpha` = 0 for pure ridge regression since we
have already determined from the custom vocabulary generation process that these features are important.
Furthermore, we use 5-fold cross-validation to obtain the best estimate of the `lambda` parameter given as `lambda_min`. To
improve training speed, the `thresh` and `maxit` arguments were set appropriately. Thus, we now have the
trained model ready for making predictions.

However, before arriving at the ideal model for our purposes, there was a lot of tinkering attempted with
tuning parameters and preprocessing.

#### 2.3.1 Tuning Model Parameters

The next steps in tuning involved adjusting the parameters for logistic regression via `cv.glmnet`. After
trying `alpha` values in the range of (0, 0.5, 1), we saw that the best results were obtained when `alpha` = 0. The `lambda`
parameter was automatically tuned via 5-fold cross-validation and `lambda_min` was used.

## 2.4 Model Evaluation

To evaluate the model, we load the test data and run the same preprocessing steps as we did on the
training data (remove special characters, tokenize and lowercase). Then, we create a DTM for the test data
and generate predictions for each review by passing it to our model. 

**Note:** Despite this being a binary
classification problem, the results are not class labels. The results in this case, for the sake of using AUC as
an evaluation metric, were probabilities of the observation to belong to class 1 (that is, the probability of
the current observation to have a positive sentiment score). The results were saved in an output file, and
then the evaluation code provided in the project details was run for each of the five splits. The criteria used
for model evaluation was AUC. The goal was to train a model with a vocabulary of less than 1000 terms such
that each split returned an AUC of 0.96 or greater on the test data.

# 3 Results

After generating a vocabulary, each split was individually tested. The model, using the same vocabulary
list, was trained on the current split’s training data, and then evaluated on the current split’s testing data.
The results are shown below.

# 3.1 Model Performance per Split (AUC)

Below, we have the AUC results for each of the five splits.

| Split | #1 | #2 | #3 | #4 | #5 |
| --- | --- | --- | --- |--- |--- |
| AUC | 0.9631 | 0.9626 | 0.9623 | 0.9633 | 0.9624 |

Each split resulted in an AUC of 0.96 or greater. Furthermore, each split used a model
trained on the same vocabulary of length 684. Thus, these results show that the requirements for the model
have been satisfied.

# 3.2 Model Accuracy per Split 

| Split | #1 | #2 | #3 | #4 | #5 |
| --- | --- | --- | --- |--- |--- |
| AUC | 0.9013 | 0.9011 | 0.9022 | 0.9002 | 0.8999 |

# 4 Conclusion

Some of the more interesting findings were related to preprocessing the data. It was quite surprising to
find that TF-IDF and removing stopwords did not improve the performance of this model. Furthermore,
reducing the size of the vocabulary from 2000 to 684 did not dramatically reduce the accuracy of the model.
When investigating the reviews that were misclassified, some patterns emerged where the review was not
particularly “harsh” and spent a lot of time describing the plot using terms that could appear in a positive
review. Another issue was detecting sarcasm, which is a key issue in all aspects of sentiment analysis and can
potentially be addressed via neural network-based approaches (or others). In summary, this current model
strikes a good balance between interpretability and accuracy. After reaching the minimum project cut-off
requirements, the size of the vocabulary was reduced while still ensuring it beat the AUC cutoffs. The reason
for this was increase interpretability; a smaller vocabulary is more easily explainable than a larger one.

Potential future improvements could include utilizing more modern methods of classification such as
LSTM-based neural networks. Such an approach can be seen described in a paper published by researchers at
Yunnan University. Neural networks, as mentioned above, could help with difficult cases such as detecting
sarcasm. Other potential improvements could involve investigating different classification methods such as
SVM which were not considered in this project because of the training time being too high compared to
logistic regression. Furthermore, as we saw in this project, increasing the training size to include the data
from all five splits drastically improved model performance. Instead of just using these 50,000 reviews, we
could potentially include hundreds of thousands of movie reviews from other sources (even IMDb itself) to
increase the overall vocabulary set, and then identifying a subset of terms from there. The way this project
was completed involved utilizing basically all the reviews to generate the vocabulary. Ideally, we would find
a way to avoid this by expanding the vocabulary via other sources and testing on completely unseen data.
Another way to potentially improve this model could be to apply stemming to reduce the sheer number of
different permutations of the same words. Lastly, we could apply more NLP techniques such as part-of-speech
tagging, feature extraction, etc., to utilize techniques that have been tailored for NLP tasks.

# 5 Sources

This project was completed as a part of CS 598 (Practical Statistical Learning) at UIUC. The sources are listed below.

1. Professor Liang's Notes

2. https://cran.r-project.org/web/packages/text2vec/vignettes/text-vectorization.html

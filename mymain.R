# Import Libraries
library(pROC)
library(text2vec)
library(glmnet)

# Set an initial seed
set.seed(6584)


## Load the (generated) vocabulary and training data
# Read in vocabulary file
myvocab <- scan(file = "myvocab.txt", what = character())

# Read and preprocess the training data
train <- read.table("train.tsv", stringsAsFactors = FALSE,
                    header = TRUE)
train$review = gsub('<.*?>', ' ', train$review)


## Train a binary classification model
# Create iterator
it_train = itoken(train$review,
                  preprocessor = tolower,
                  tokenizer = word_tokenizer)

# Create vocabulary from vocabulary file
new_vocab = create_vocabulary(myvocab,
                              ngram = c(1L, 2L))

# Create Training Document-Term-Matrix (DTM)
vectorizer = vocab_vectorizer(new_vocab)
dtm_train = create_dtm(it_train, vectorizer)

# Fit a logistic regression model to the DTM and sentiment scores
NFOLDS = 4
glmnet_classifier <- cv.glmnet(
  x = dtm_train,
  y = train$sentiment,
  family = "binomial",
  # L1 penalty
  alpha = 0, # ridge
  # Interested in Area Under ROC curve
  type.measure = "auc",
  # 5-fold cross-validation
  nfolds = NFOLDS,
  # high value is less accurate, but has faster training
  thresh = 1e-3,
  # again lower number of iterations for faster training
  maxit = 1e3
)

## Use the model to predict the sentiments of the test data
# Read in test data and preprocess it
test <- read.table("test.tsv", stringsAsFactors = FALSE,
                   header = TRUE)
test$review = gsub('<.*?>', ' ', test$review)

# Create iterator
it_test = itoken(
  test$review,
  ids = test$id,
  preprocessor = tolower,
  tokenizer = word_tokenizer
)

# Create Test DTM
dtm_test = create_dtm(it_test, vectorizer)

# Generate Predictions
preds = predict(glmnet_classifier, dtm_test, s = "lambda.min", type = 'response')[, 1]
output = data.frame(id = test$id, prob = as.vector(preds))

# Predict Accuracy (Uncomment only to generate results)
# preds_class = predict(glmnet_classifier, dtm_test, s = "lambda.min", type = 'class')[, 1]
# tb = table(actual = test.y$sentiment, predicted = preds_class)
# print(1 - ((tb[2] + tb[3]) / (tb[2] + tb[3] + tb[1] + tb[4])))

#####################################
# Compute prediction
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predited probabilities
#####################################
write.table(output,
            file = "mysubmission.txt",
            row.names = FALSE,
            sep = '\t')
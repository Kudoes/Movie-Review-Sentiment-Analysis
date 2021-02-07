## This code generates a vocabulary of words that is as small as possible
## while still ensuring the ROC remains above 0.95.

# Import Libraries
library(pROC)
library(text2vec)
library(slam)
library(glmnet)

# Set seed
set.seed(6584)

## Initial Vocab. Construction ##
# Read in the full training data from all 5 splits
train = read.table("alldata.tsv",
                   stringsAsFactors = FALSE,
                   header = TRUE)

# Remove special characters
train$review = gsub('<.*?>', ' ', train$review)

# Create iterator over tokens while specify the preprocessing and tokenization functions
it_train = itoken(train$review,
                  ids = train$id,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)

# Create the vocabulary using the iterator
tmp.vocab = create_vocabulary(it_train,
                              ngram = c(1L,4L))

# Prune the vocabulary of uncommon words
tmp.vocab = prune_vocabulary(tmp.vocab, 
                             term_count_min = 25,
                             doc_proportion_max = 0.5,
                             doc_proportion_min = 0.001)

# Now, using the created vocab, create a Doc. Term Matrix
vectorizer = vocab_vectorizer(tmp.vocab)
dtm_train  = create_dtm(it_train, vectorizer)

## Two-Sample T-Test ##
v.size = dim(dtm_train)[2]
ytrain = train$sentiment

summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), mean)
summ[,2] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), var)
summ[,3] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), mean)
summ[,4] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), var)

n1 = sum(ytrain); 
n = length(ytrain)
n0 = n - n1

myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0)

# Identify top 2000 terms and save them
words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:2000]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]
words_sel = words[id]
it_train = itoken(train$review,
                  preprocessor = tolower,
                  tokenizer = word_tokenizer)

# Specify new vocabulary
new_vocab = create_vocabulary(words_sel,
                              ngram = c(1L, 4L))

# Create the vocabulary using the iterator
vectorizer = vocab_vectorizer(new_vocab)
dtm_train = create_dtm(it_train, vectorizer)

## LASSO for Subset Selection of Terms ##
# Train logistic regression model of vocab against sentiments
tmpfit = glmnet(x = dtm_train, 
                y = train$sentiment,
                alpha = 1,
                family = 'binomial')

# Identify the top 700 words and save in a list
ideal_idx = max(which(tmpfit$df < 700))
myvocab = colnames(dtm_train)[which(tmpfit$beta[, ideal_idx] != 0)]

# Write the vocabulary to the directory
write.table(myvocab,
            "myvocab.txt",
            row.names = FALSE,
            col.names = FALSE)
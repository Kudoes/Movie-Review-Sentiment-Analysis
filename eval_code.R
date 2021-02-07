## This code computes the ROCs for all splits of test data.

# Import Library
library(pROC)

# Load vocab in main dir
myvocab = read.table("myvocab.txt",
                     stringsAsFactors = FALSE,
                     header = TRUE)
system.time({
  for (j in 1:5) {
    set.seed(6584)
    print(paste("Split: ", j))
    train = read.table(
      paste("split_", j, "/train.tsv", sep = ""),
      stringsAsFactors = FALSE,
      header = TRUE
    )
    test = read.table(
      paste("split_", j, "/test.tsv", sep = ""),
      stringsAsFactors = FALSE,
      header = TRUE
    )
    test.y = read.table(
      paste("split_", j, "/test_y.tsv", sep = ""),
      stringsAsFactors = FALSE,
      header = TRUE
    )
    
    source("mymain.R")
    
    pred <- read.table("mysubmission.txt", header = TRUE)
    pred <- merge(pred, test.y, by = "id")
    roc_obj <- roc(pred$sentiment, pred$prob)
    print(pROC::auc(roc_obj))
  }
})

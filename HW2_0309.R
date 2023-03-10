
# libraries
library(dplyr)
library(tidyverse)
library(ggplot2)

library(sentimentr)

library(quanteda)
library(quanteda.textmodels)
library(quanteda.textplots)
library(quanteda.textstats)

library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)

setwd("C:/Users/younskim/Documents/TextAsData/HW2")
list.files()

base::load("russian_trolls_sample.rdata")
View(russian_trolls_sample)

# copurs
trolls_corp = corpus(russian_trolls_sample)
docvars(trolls_corp) %>% head(2)
trolls_corp[1]

# EDA
trolls_dfm = trolls_corp %>% 
  tokens(remove_punct = T, remove_symbols = T, remove_numbers = T, remove_url = T) %>% 
  tokens_tolower() %>% 
  tokens_wordstem("en") %>% 
  tokens_select(stopwords("en"), selection="remove") %>% 
  tokens_ngrams(1) %>% 
  dfm()

topfeatures(trolls_dfm, 20, groups=account_category)



"""
reference for word choice:
https://www.brookings.edu/blog/fixgov/2015/06/03/republicans-and-democrats-divided-on-important-issues-for-a-presidential-nominee/
https://www.pewresearch.org/fact-tank/2019/02/05/republicans-and-democrats-have-grown-further-apart-on-what-the-nations-top-priorities-should-be/
"""

R_my_dict = "budget deficit national defense tax terrorism job creation immigration trade economy military crime social security white supremacy makes america great again Trump supporters fake news"


CleanDictionary = function(x){
  x = tokens(x,  remove_numbers = TRUE, remove_punct = TRUE)
  x = tokens_wordstem(x)
  x = tokens_select(x, pattern = stopwords('en'), selection = 'remove')
  x = tokens_tolower(x)
  return(as.character(x)) 
}

my_cl_dict = CleanDictionary(R_my_dict)
my_cl_dict


# right trolls on the issue
table(trolls_dfm$account_category) # L=5112, R=4888

# right_total_words = 43785
right_total_words = sum(trolls_dfm[docvars(trolls_dfm)$account_category == "RightTroll",])

# right_issue_words = 2370
right_issue_words = sum(trolls_dfm[docvars(trolls_dfm)$account_category == "RightTroll", colnames(trolls_dfm) %in% my_cl_dict])

# about 3% of proportion 
right_prop_issue = right_issue_words / right_total_words
right_prop_issue*100



# left_total_words = 45865
left_total_words = sum(trolls_dfm[docvars(trolls_dfm)$account_category == "LeftTroll",])

# right_issue_words = 1097
left_issue_words = sum(trolls_dfm[docvars(trolls_dfm)$account_category == "LeftTroll", colnames(trolls_dfm) %in% my_cl_dict])

# about 2% of proportion 
left_prop_issue = left_issue_words / left_total_words
left_prop_issue*100


# results
results = data.frame(account_category = c("Right", "Left"), 
                     prop_on_issue = c(right_prop_issue, left_prop_issue))
results %>% head(3)


ggplot(results, aes(x=account_category, y=prop_on_issue, fill=account_category))+
  geom_bar(stat="identity") + theme_minimal() + 
  xlab("Account Cagegory") + 
  ylab("Attention to issue (proportion)")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+scale_fill_manual(values=c("blue", "red"))

# time variation
results = data.frame(account_category = c("Right", "Left"), 
                     prop_on_issue = c(right_prop_issue, left_prop_issue),
                     date = docvars(trolls_corp)$date)
results %>% head(3)
ggplot(results, aes(x=date, y=prop_on_issue, group=account_category, color=account_category))+
  geom_line()


# trial


trolls_dfm %>% group_by(date) %>%  sum(trolls_dfm[docvars(trolls_dfm)$account_category == "LeftTroll", colnames(trolls_dfm) %in% my_cl_dict])

docvars(trolls_dfm) %>% head(1)

##############
# sentiment analysis
###########

trolls_text = get_sentences(trolls_corp)

trolls_sentiment = sentiment_by(trolls_text, by=trolls_corp$account_category)
df_trolls_sentiment = as.data.frame(trolls_sentiment)
df_trolls_sentiment

trolls_sentiment %>% ggplot(aes(x=account_category, y=ave_sentiment, fill=account_category))+geom_bar(stat="identity")+theme_minimal()


trolls_profanity = profanity_by(trolls_text, by=trolls_corp$account_category)
trolls_profanity
trolls_profanity %>% ggplot(aes(x=account_category, y=ave_profanity, fill=account_category))+geom_bar(stat="identity")+theme_minimal()


trolls_emotion = emotion_by(trolls_text, by=trolls_corp$account_category)

trolls_3_emotion = trolls_emotion %>% filter(emotion_type %in% c("anger", "fear", "joy"))
trolls_3_emotion

trolls_3_emotion %>% ggplot(aes(x=emotion_type, y=ave_emotion, fill=account_category)) + geom_bar(stat="identity", position="dodge")

trolls_emotion %>% ggplot(aes(x=emotion_type, y=ave_emotion, fill=account_category)) + geom_bar(stat="identity", position="dodge")+coord_flip()


##################
# Machine Learning
###############

docvars(trolls_corp)
ndoc(trolls_corp)

# make a id_numeric column
docvars(trolls_corp, "id_numeric") = 1:ndoc(trolls_corp)
head(docvars(trolls_corp))

set.seed(309)

# number for training
id_train = sample(x=1:nrow(russian_trolls_sample), 7000, replace = FALSE) #train=7000, test=3000
id_train

# create dfm and preprocess it
dfm_training = corpus_subset(x= trolls_corp, subset = id_numeric %in% id_train) %>%
  tokens(remove_numbers = TRUE, remove_punct = TRUE, remove_url = TRUE,
         remove_symbols = TRUE) %>% 
  tokens_wordstem(language = "en") %>% 
  tokens_tolower() %>%
  tokens_select(pattern = stopwords('en'), selection = 'remove') %>%
  dfm() 
dim(dfm_training)

dfm_testing= corpus_subset(x= trolls_corp, subset = !id_numeric %in% id_train) %>%
  tokens(remove_numbers = TRUE, remove_punct = TRUE, remove_url = TRUE,
         remove_symbols = TRUE) %>% 
  tokens_wordstem(language = "en") %>% 
  tokens_tolower() %>%
  tokens_select(pattern = stopwords('en'), selection = 'remove') %>%
  dfm() 
dim(dfm_testing)

df_train_test = data.frame(row.names = c("train", "test"),
                           rows = c(dim(dfm_training)[1], dim(dfm_testing)[1]),
                           features = c(dim(dfm_training)[2], dim(dfm_testing)[2]))
df_train_test

# traimming
dfm_training_trim = dfm_training %>% dfm_trim(min_termfreq=5, max_termfreq = 400)
dfm_testing_trim = dfm_testing %>% dfm_trim(min_termfreq=5, max_termfreq = 400)
dim(dfm_training_trim)
dim(dfm_testing_trim)
df_train_test_trim = data.frame(row.names = c("train", "test"),
                           rows = c(dim(dfm_training_trim)[1], dim(dfm_testing_trim)[1]),
                           features = c(dim(dfm_training_trim)[2], dim(dfm_testing_trim)[2]))
df_train_test_trim



# naive bayes model
nb_classification = textmodel_nb(x = dfm_training, y = docvars(dfm_training, "account_category"))

dfm_matched = dfm_match(dfm_testing, features = featnames(dfm_training))

dfm_matched
dim(dfm_matched)
dim(dfm_testing)
dim(dfm_training)

predicted_class = predict(nb_classification, newdata = dfm_matched)
actual_class = docvars(dfm_matched, "account_category")

tab_class = table(predicted_class, actual_class)
tab_class

cm = confusionMatrix(tab_class, positive="RightTroll")
cm

nb_ac = cm$overall["Accuracy"]
nb_pr = cm$byClass["Precision"]
nb_rc = cm$byClass["Recall"]
nb_f1 = cm$byClass["F1"]


# tree model
df_training = convert(dfm_training, to="data.frame")
df_testing = convert(dfm_matched, to="data.frame")

df_training %>% head(1)
df_testing %>% head(1)


df_training$account_category <- as.factor(docvars(dfm_training, "account_category"))

tree_classification = rpart(formula = account_category ~ ., data=df_training[,!colnames(df_training) == "doc_id"])

predicted_class = predict(tree_classification, newdata = df_testing[,!colnames(df_testing) == "doc_id"], type="class")
actual_class = docvars(dfm_matched, "account_category")

tab_tree = table(predicted_class, actual_class)
confusionMatrix(tab_tree, positive="RightTroll")

cm_tree = confusionMatrix(tab_tree, mode = "everything", positive = "RightTroll")


tr_ac = cm_tree$overall["Accuracy"]
tr_pr =cm_tree$byClass["Precision"]
tr_rc = cm_tree$byClass["Recall"]
tr_f1= cm_tree$byClass["F1"]

nb_ac = cm$overall["Accuracy"]
nb_pr = cm$byClass["Precision"]
nb_rc = cm$byClass["Recall"]
nb_f1 = cm$byClass["F1"]

df_results = data.frame(row.names=c("nb", "tree"),
                        accuracy = c(nb_ac, tr_ac),
                        precision = c(nb_pr, tr_pr),
                        recall = c(nb_rc, tr_rc),
                        f1_score = c(nb_f1, tr_f1))
df_results

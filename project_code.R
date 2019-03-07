# --------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(neuralnet)
library(randomForest)
library(gbm)
library(e1071)
library(pROC)
library(caret)
library(cvTools)

# ------------------------

bdtr<- read.csv("E:\\edvancer\\business_analytics\\r_programming\\upgrad\\bank-additional\\bank-additional-full.csv",
               stringsAsFactors = F,sep = ";")
bdts <- read.csv("E:\\edvancer\\business_analytics\\r_programming\\upgrad\\bank-additional\\bank-additional.csv",
                stringsAsFactors = F,sep = ";")

bdtr <- bdtr %>% 
  mutate(data="train")

bdts <- bdts %>% 
  mutate(data="test")

all_bd <- rbind(bdtr,bdts)


summary(all_bd)
glimpse(all_bd)
# --------------------------------

# mode of categorical variable
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# ------------------------------
names(all_bd) # name of columns
# missing values 
for (i in names(all_bd)) {
  print(paste(i,":",sum(is.na(all_bd[i]))))
}

# ---------------------
# TO create Dummy Variables 

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

# ------------------------
# To check Categorical variables or features

col_names<-names(all_bd[sapply(all_bd,function(x) is.character(x))])
col_names


# ----------------------------------------------------

# CATEGORICAL VARIABLES CONVERSION TO DUMMIES

#  JOB
sort(table(all_bd$job),decreasing = T)
all_bd = CreateDummies(all_bd,"job",1000)

# MARITAL
sort(table(all_bd$marital),decreasing = T)
all_bd = CreateDummies(all_bd,"marital",5000)

# EDUCATION
sort(table(all_bd$education),decreasing = T)
all_bd = CreateDummies(all_bd,"education",2500)

# DEFAULT
sort(table(all_bd$default),decreasing = T)
all_bd = CreateDummies(all_bd,"default",5000)

# HOUSING
sort(table(all_bd$housing),decreasing = T)
all_bd = CreateDummies(all_bd,"housing",5000)

# LOAN
sort(table(all_bd$loan),decreasing = T)
all_bd = CreateDummies(all_bd,"loan",5000)

# CONTACT
sort(table(all_bd$contact),decreasing = T)
all_bd = CreateDummies(all_bd,"contact",0)

# MONTH
sort(table(all_bd$month),decreasing = T)
all_bd = CreateDummies(all_bd,"month",100)

# DAY OF WEEK
sort(table(all_bd$day_of_week),decreasing = T)
all_bd = CreateDummies(all_bd,"day_of_week",0)

# POUTCOME
sort(table(all_bd$poutcome),decreasing = T)
all_bd = CreateDummies(all_bd,"poutcome",1000)
# ------------------------

cor_mat <- round(cor(all_bd[c(1:10)]),2)
library(reshape2)
melted_cormat = melt(cor_mat)

ggplot(melted_cormat,aes(x=Var1,y=Var2,fill=value)) + geom_tile()
# -------------------------------------

# AGE - NUMERIC
hist(all_bd$age)
boxplot(all_bd$age)



# DURATION - NUMERIC
hist(log(all_bd$duration))
boxplot(log(all_bd$duration))
# -------------------------------

# CAMPAIGN - CATEGORICAL
sort(table(all_bd$campaign),decreasing = T)
all_bd = CreateDummies(all_bd,"campaign",100)

# PDAYS - CAETGORICAL
sort(table(all_bd$pdays),decreasing = T)
all_bd = CreateDummies(all_bd,"pdays",100)

# PREVIOUS - CATEGORICAL
sort(table(all_bd$previous),decreasing = T)
all_bd = CreateDummies(all_bd,"previous",100)

# -------------------------------------
all_bd<-all_bd %>%
  mutate(age=as.numeric(scale(all_bd$age)),       
         duration=as.numeric(scale(all_bd$duration)),   
         emp.var.rate=as.numeric(scale(all_bd$emp.var.rate)),
         cons.price.idx=as.numeric(scale(all_bd$cons.price.idx)), 
         cons.conf.idx=as.numeric(scale(all_bd$cons.conf.idx)),
         euribor3m=as.numeric(scale(all_bd$euribor3m)),
         nr.employed=as.numeric(scale(all_bd$nr.employed)))
# -----------------------------------
# TARGET VARIABLE "Y"

all_bd$y = as.numeric(all_bd$y=="yes")
all_bd$y = as.factor(all_bd$y)

# -----------------------------------

# PREPARED TRAINING DATASET
bd_train = all_bd[which(all_bd$data=="train"),]
View(bd_train)
bd_train = bd_train %>% select(-data)
View(bd_train)

# PREPARED TEST DATASET
bd_test = all_bd[which(all_bd$data=="test"),]
View(bd_test)
bd_test = bd_test %>% select(-data)

# -------------------------------------
# MODEL BUILDING

subset_paras = function(full_list_paras,n = 10){
  all_comb = expand.grid(full_list_paras)
  s = sample(1:nrow(all_comb),n)
  subset_para = all_comb[s,]
  return(subset_para)
}

mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(score)
}

param=list(mtry=c(5,10,15,20,25,35),
           ntree=c(50,100,200,500,700),
           maxnodes=c(5,10,15,20,30,50,100),
           nodesize=c(1,2,5,10)
)


num_trials=50
my_params=subset_paras(param,num_trials)
my_params

myauc=0

## Cvtuning
## This code will take couple hours to finish
## Dont execute in the class
for(i in 1:num_trials){
  print(paste('starting iteration :',i))
  # uncomment the line above to keep track of progress
  params=my_params[i,]

  k=cvTuning(randomForest,y~.,
             data =bd_train,
             tuning =params,
             folds = cvFolds(nrow(bd_train), K=10, type ="random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="prob")
  )
  score.this=k$cv[,2]

  if(score.this>myauc){
    #print(params)
    # uncomment the line above to keep track of progress
    myauc=score.this
    print(myauc)
    # uncomment the line above to keep track of progress
    best_params=params
  }

  #print('DONE')
  # uncomment the line above to keep track of progress
}



final_model <- randomForest(y~.,
                            data = bd_train,
                            mtry = 40,
                            ntree = 700,
                            maxnodes = 100,
                            nodesize = 20,
                            do.trace = T)
final_model


pred<-round(predict(final_model,
                    newdata = bd_test[-8],
                    type = "prob")[,2])
pred

pred_df <- as.data.frame(pred)
names(pred_df) <- "y"

write.csv(final_rf_pred,"submission.csv",
          row.names = F)


roc(bd_train$y,pred)

confusionMatrix(as.factor(pred),bd_test$y)
pROC::auc(bd_test$y,pred)



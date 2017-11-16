indian.data <- read.table("~/pima-indians-diabetes.csv",sep = ",")
dim(indian.data)
Y <- as.matrix(indian.data[ ,9], ncol=1)
X <- as.matrix(indian.data[ ,-9],ncol=8)
#Y.train <- Y[1:600, ]
#X.train <-as.matrix(X[1:600, ], ncol=8)
#Y.test <- Y[601:768, ]
#X.test <- data.frame(X[601:768,])


logitfit <- glm(Y~X,family=binomial(link='logit'))

fitted.results <- predict(logitfit,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != Y)
print(paste('Accuracy',1-misClasificError))



rejoin

gabor [12:17]
butyou know what, let's try to run linear regression too, and see what comes out


bartek
[12:17]
can you see im trying to call you ?


gabor [12:17]
no,lk

bartek
[12:17]
Started a call


gabor [12:22]
indian.data <- read.table("~/pima-indians-diabetes.csv",sep = ",")
dim(indian.data)
Y <- as.matrix(indian.data[ ,9], ncol=1)
X <- as.matrix(indian.data[ ,-9],ncol=8)
#Y.train <- Y[1:600, ]
#X.train <-as.matrix(X[1:600, ], ncol=8)
#Y.test <- Y[601:768, ]
#X.test <- data.frame(X[601:768,])

olsfit <- lm(Y~X)
summary(olsfit)

#logitfit <- glm(Y~X,family=binomial(link='logit'))
#summary(logitfit)

fitted.results <- predict(olsfit,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != Y)
print(paste('Accuracy',1-misClasificError))


gabor
[12:32]
x.cons <- cbind(1,X)


[12:32]
beta <- solve(t(x.cons) %*% x.cons) %*% t(x.cons) %*% Y

[12:32]
(solve is matrix inverse)
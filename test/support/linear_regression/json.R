library(jsonlite)

x <- 1:5
y <- 5*x + 3
data <- data.frame(x=x, y=y)

model <- lm(y ~ x, data)
cat(toJSON(list(coefficients=as.list(coef(model))), auto_unbox=TRUE))

library(pmml, quiet=TRUE)

x <- 1:5
y <- 5*x + 3
data <- data.frame(x=x, y=y)

model <- lm(y ~ x, data)
cat(toString(pmml(model)))

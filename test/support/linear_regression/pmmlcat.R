library(pmml, quiet=TRUE)

x <- 1:4
weekday <- c("Sunday", "Sunday", "Monday", "Monday")
y <- c(12, 14, 22, 24)
data <- data.frame(x=x, weekday=weekday, y=y)

model <- lm(y ~ ., data)
cat(toString(pmml(model)))

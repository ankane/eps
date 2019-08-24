library(aurelius)

x <- 1:4
weekday <- c("Sunday", "Sunday", "Monday", "Monday")
y <- c(12, 14, 22, 24)
data <- data.frame(x=x, weekday=weekday, y=y)

model <- lm(y ~ ., data)
write_pfa(pfa(model))

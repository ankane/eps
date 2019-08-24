library(aurelius)

x <- 1:5
y <- 5*x + 3
data <- data.frame(x=x, y=y)

model <- lm("y ~ x", data)
write_pfa(pfa(model))

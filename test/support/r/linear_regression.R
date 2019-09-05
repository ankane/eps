library(pmml, quiet=TRUE)

data <- read.csv("test/support/mpg.csv")

model <- lm(hwy ~ displ + year + cyl + drv + class, data)

pmml <- toString(pmml(model))

write(pmml, file="test/support/r/linear_regression.pmml")

print(predict(model, data[0:10,]))

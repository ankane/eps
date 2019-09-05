library(pmml, quiet=TRUE)
library(e1071)

data <- read.csv("test/support/mpg.csv")
model <- naiveBayes(drv ~ displ + year + cyl + class, data, laplace=1)

pmml <- toString(pmml(model, predicted_field="drv"))
write(pmml, file="test/support/r/naive_bayes.pmml")

print(predict(model, data[0:10,]))

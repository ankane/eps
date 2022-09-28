## 0.4.1 (2022-09-28)

- Fixed `cannot load such file -- matrix` error with Ruby 3.1

## 0.4.0 (2022-09-02)

- Fixed `stack level too deep` error with many rows
- Dropped support for `gsl` gem (use `gslr` instead)
- Dropped support for Ruby < 2.7

## 0.3.9 (2021-10-14)

- Fixed error with `lessOrEqual` operator

## 0.3.8 (2021-02-08)

- Fixed error with categorical and text features

## 0.3.7 (2020-11-23)

- Fixed error with LightGBM summary

## 0.3.6 (2020-06-19)

- Fixed error with text features for LightGBM

## 0.3.5 (2020-06-10)

- Added `learning_rate` option for LightGBM
- Added support for Numo and Rover

## 0.3.4 (2020-04-05)

- Added `predict_probability` for classification

## 0.3.3 (2020-02-24)

- Fixed errors and incorrect predictions with boolean columns
- Fixed deprecation warnings in Ruby 2.7

## 0.3.2 (2019-12-08)

- Added support for GSLR

## 0.3.1 (2019-12-06)

- Added `weight` option for LightGBM and linear regression
- Added `intercept` option for linear regression
- Added LightGBM evaluator safety check
- Fixed `Unknown label` error for LightGBM
- Fixed error message for unstable solutions with linear regression

## 0.3.0 (2019-09-05)

- Added support for LightGBM
- Added text features
- Fixed naive Bayes PMML
- Fixed error with classification and Daru

Breaking

- LightGBM is now the default for new models
- Cross-validation happens automatically by default
- Removed support for JSON and PFA formats
- Added smoothing to naive Bayes

## 0.2.1 (2019-05-19)

- Fixed error with `summary`
- Fixed error with `predict` in `Eps::Base`
- Fixed error with loaded classification models

## 0.2.0 (2019-05-19)

- Added support for classification
- Added `to_pmml` method
- Added `Eps::Base`

## 0.1.1 (2018-07-05)

- Huge performance boost

## 0.1.0 (2018-07-03)

- First release

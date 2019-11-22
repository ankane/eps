## 0.3.1 (unreleased)

- Added `weight` option for LightGBM and linear regression
- Added `intercept` option for linear regression

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

source "https://rubygems.org"

git_source(:github) {|repo_name| "https://github.com/#{repo_name}" }

# Specify your gem's dependencies in eps.gemspec
gemspec

gem "gsl" if ENV["GSL"]
gem "onnxruntime", path: "~/open_source/onnxruntime"

source "https://rubygems.org"

git_source(:github) {|repo_name| "https://github.com/#{repo_name}" }

# Specify your gem's dependencies in eps.gemspec
gemspec

# remove when 0.2.1 released
gem "daru", github: "sciruby/daru"

gem "gsl" if ENV["GSL"]

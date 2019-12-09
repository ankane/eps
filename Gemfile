source "https://rubygems.org"

git_source(:github) {|repo_name| "https://github.com/#{repo_name}" }

# Specify your gem's dependencies in eps.gemspec
gemspec

if ENV["GSL"] == "gslr"
  gem "gslr", ">= 0.1.2"
elsif ENV["GSL"]
  gem "gsl"
end

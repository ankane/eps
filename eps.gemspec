require_relative "lib/eps/version"

Gem::Specification.new do |spec|
  spec.name          = "eps"
  spec.version       = Eps::VERSION
  spec.summary       = "Machine learning for Ruby. Supports regression (linear regression) and classification (naive Bayes)"
  spec.homepage      = "https://github.com/ankane/eps"
  spec.license       = "MIT"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.7"

  spec.add_dependency "lightgbm", ">= 0.1.7"
  spec.add_dependency "matrix"
  spec.add_dependency "nokogiri"
end

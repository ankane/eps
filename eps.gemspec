require_relative "lib/eps/version"

Gem::Specification.new do |spec|
  spec.name          = "eps"
  spec.version       = Eps::VERSION
  spec.summary       = "Machine learning for Ruby. Supports regression (linear regression) and classification (naive Bayes)"
  spec.homepage      = "https://github.com/ankane/eps"
  spec.license       = "MIT"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@chartkick.com"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.4"

  spec.add_dependency "lightgbm", ">= 0.1.7"
  spec.add_dependency "nokogiri"

  spec.add_development_dependency "bundler"
  spec.add_development_dependency "daru"
  spec.add_development_dependency "minitest"
  spec.add_development_dependency "rake"
end

require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
  t.warning = false
end

task default: :test

task :benchmark do
  require "benchmark"
  require "eps"
  # require "gslr"

  data = []
  10000.times do
    row = {}
    30.times do |i|
      row[:"x#{i}"] = rand(100)
    end
    row[:y] = rand(100)
    data << row
  end

  puts "Starting benchmark..."

  time = Benchmark.realtime do
    Eps::Regressor.new(data, target: :y)
  end
  p time.round(1)
end

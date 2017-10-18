require 'awesome_print'
require 'matrix'

require_relative 'data'

# Monkey patchs
# ----------------------------------------------
# 1 / 1 + et
class Float
  def sigmoid
    1.0 / (1 + Math.exp(-self))
  end
end

class Array
  def to_v
    Vector.elements(self)
  end

  def sum
    inject(:+)
  end
end

# Neural network
# ----------------------------------------------
class Neuron
  LEARNING_RATE = 0.2
  def initialize(inputs_count)
    @weights = Array.new(inputs_count) { rand }.to_v
  end

  def forward(inputs)
    @inputs = inputs
    inputs.dot(@weights).sigmoid
  end

  def back_propagate(error)
    @delta = error * LEARNING_RATE
    errors = @weights * @delta
    update_weights
    errors
  end

  private

  def update_weights
    @weights -= @inputs * @delta
  end
end

class Layer
  def initialize(neurons_count, inputs_count)
    @neurons = Array.new(neurons_count) { Neuron.new(inputs_count) }
  end

  def forward(inputs)
    @neurons.map { |neuron| neuron.forward(inputs) }.to_v
  end

  def back_propagate(errors)
    @neurons.zip(errors).map do |neuron, error|
      neuron.back_propagate(error)
    end.sum
  end
end

class Network
  def initialize(*layers)
    @layers = layers
  end

  def forward(inputs)
    @layers.inject(inputs) do |inputs, layer|
      layer.forward(inputs)
    end
  end

  def back_propagate(errors)
    @layers.reverse.each do |layer|
      errors = layer.back_propagate(errors)
    end
  end
end

# Useful output stuff
# ----------------------------------------------

def print_prediction(outputs)
  puts "A: #{(outputs[0].round(2) * 100).to_i}%"
  puts "B: #{(outputs[1].round(2) * 100).to_i}%"
  puts "C: #{(outputs[2].round(2) * 100).to_i}%"
  puts "D: #{(outputs[3].round(2) * 100).to_i}%"
  puts "E: #{(outputs[4].round(2) * 100).to_i}%"
  puts "F: #{(outputs[5].round(2) * 100).to_i}%"
end

def print_letter(letter)
  puts ""
  20.times do |i|
    print "X" if letter[i] == 1
    print " " if letter[i] == 0
    puts "" if ((i+1) % 4) == 0
  end
  puts ""
end

# Testing
# ----------------------------------------------
nn = Network.new(
  Layer.new(20, 20),
  Layer.new(10, 20),
  Layer.new(6, 10)
)

# training

TRAINING_DATA = [
  { letter: LETTER_A.to_v, expected: EXPECTED_A.to_v },
  { letter: LETTER_B.to_v, expected: EXPECTED_B.to_v },
  { letter: LETTER_C.to_v, expected: EXPECTED_C.to_v },
  { letter: LETTER_D.to_v, expected: EXPECTED_D.to_v },
  { letter: LETTER_E.to_v, expected: EXPECTED_E.to_v },
  { letter: LETTER_F.to_v, expected: EXPECTED_F.to_v },
];

1000.times do
  TRAINING_DATA.each do |sample|
    output = nn.forward(sample[:letter])
    errors = output - sample[:expected]
    nn.back_propagate(errors)
  end
end

# test data

ap "---------------- A --------------------"
output = nn.forward(LETTER_A.to_v)
print_prediction(output)
ap "---------------------------------------"

ap "---------------- B --------------------"
output = nn.forward(LETTER_B.to_v)
print_prediction(output)
ap "---------------------------------------"

ap "---------------- C --------------------"
output = nn.forward(LETTER_C.to_v)
print_prediction(output)
ap "---------------------------------------"

ap "---------------- D --------------------"
output = nn.forward(LETTER_D.to_v)
print_prediction(output)
ap "---------------------------------------"

ap "---------------- E --------------------"
output = nn.forward(LETTER_E.to_v)
print_prediction(output)
ap "---------------------------------------"

ap "---------------- F --------------------"
output = nn.forward(LETTER_F.to_v)
print_prediction(output)
ap "---------------------------------------"

letter = [
  1, 1, 1, 0,
  1, 0, 0, 0,
  1, 0, 0, 0,
  1, 0, 0, 0,
  1, 1, 1, 1
];

print_letter(letter)
ap "---------------- ? --------------------"
output = nn.forward(letter.to_v)
print_prediction(output)
ap "---------------------------------------"

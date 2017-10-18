require 'awesome_print'
require 'matrix'

require_relative 'data'

# 1 / 1 + et
def sigmoid(t)
  1.0 / (1 + Math.exp(-t))
end

class Neuron
  def initialize(inputs_count)
    @weights = Array.new(inputs_count) { rand }.to_v
  end

  def forward(inputs)
    @inputs = inputs
    sigmoid(inputs.dot(@weights))
  end

  def back_propagate(error)
    @delta = error * 0.2
    errors = @weights * @delta
    # update weights
    @weights -= @inputs * @delta
    errors
  end
end

class Layer
  def initialize(neurons)
    @neurons = neurons
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
  def initialize(layers)
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

class Array
  def to_v
    Vector.elements(self)
  end

  def sum
    inject(:+)
  end
end

def print_prediction(outputs)
  puts "A: #{(outputs[0].round(2) * 100).to_i}%"
  puts "B: #{(outputs[1].round(2) * 100).to_i}%"
  puts "C: #{(outputs[2].round(2) * 100).to_i}%"
  puts "D: #{(outputs[3].round(2) * 100).to_i}%"
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

layers = []

# first layer 4 * 5 = 20 neurons
neurons = (1..20).to_a.map { |x| Neuron.new(20) }
first_layer = Layer.new(neurons)
layers << first_layer

# second layer = 10 neurons
neurons = (1..10).to_a.map { |x| Neuron.new(20) }
second_layer = Layer.new(neurons)
layers << second_layer

# third layer = 4 neurons (A,B,C,D)
neurons = (1..4).to_a.map { |x| Neuron.new(10) }
output_layer = Layer.new(neurons)
layers << output_layer

nn = Network.new(layers)

# training

1000.times do
  output = nn.forward(LETTER_A.to_v)
  errors = output - EXPECTED_A.to_v
  nn.back_propagate(errors)

  output = nn.forward(LETTER_B.to_v)
  errors = output - EXPECTED_B.to_v
  nn.back_propagate(errors)

  output = nn.forward(LETTER_C.to_v)
  errors = output - EXPECTED_C.to_v
  nn.back_propagate(errors)

  output = nn.forward(LETTER_D.to_v)
  errors = output - EXPECTED_D.to_v
  nn.back_propagate(errors)
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


include("components.jl")

using IfElse

# Gate models (activation, inactivation, etc) for voltage-gated membrane currents
# Constructors could be adjusted, made nicer, and/or convenient macro syntax made up for this
n = AlphaBeta(
    V -> IfElse.ifelse(V == -55.0, 0.1 , (0.01*(V + 55.0))/(1.0 - exp(-(V + 55.0)/10.0))),
    V -> 0.125 * exp(-(V + 65.0)/80.0),
    exponent = 4)

m = AlphaBeta(
    V -> IfElse.ifelse(V == -40.0, 1.0, (0.1*(V + 40.0))/(1.0 - exp(-(V + 40.0)/10.0))),
    V -> 4.0*exp(-(V + 65.0)/18.0),
    exponent = 3)

# h = AlphaBeta(
#     V -> 0.07*exp(-(V+65.0)/20.0),
#     V -> 1.0/(1.0 + exp(-(V + 35.0)/10.0)))
# We can also define using steady-state (g∞) and time constant (τ) as in NeuronBuilder:
h = SSTau(
    V -> 1.0/(1.0+exp((V+12.3)/-11.8)),
    V -> 7.2 - 6.4/(1.0+exp((V+28.3)/-19.2)))

# Synaptic conductances fall in the range of 10-100nS
@named FastNaV = VoltageGatedChannel{Na}([m, h], max_g = 120mS/cm^2)
@named DelayedRect = VoltageGatedChannel{K}([n], max_g = 36mS/cm^2)
@name KCaV = VoltageIonGatedChannel{K,Ca}(
MembraneLeak = PassiveChannel{Leak}(max_g = 0.3mS/cm^2)

membrane_currents = [FastNaV, DelayedRect, Leak]

eq_potentials = [Na => 50mV,
                 K  => -80mV,
                 Leak => -50mV]

# And continue on with forms similar to this...
@named SomeNeuron = Neuron(membrane_currents, eq_potentials)

@named MyCircuit = NeuronalNetwork(Vector{Neurons}, 



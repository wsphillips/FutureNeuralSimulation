# This is an example of two synaptically-coupled single-compartment neurons with Hodgkin-Huxley dynamics.
# It's written for illustrative purposes and uses the high-level interface of ModelingToolkit.jl as it already exists today.

using DifferentialEquations, ModelingToolkit, OrdinaryDiffEq, Unitful

import Unitful: Voltage

# Channel gating functions
function αm(V::Voltage)
    V = ustrip(Float64, u"mV", V)    
    return (0.01*(V + 55))/(1 - exp(-(V + 55)/10))
end


function βn(V::Voltage)
    V = ustrip(Float64, u"mV", V)    
    return .125 * exp(-(V + 65)/80)
end


function αm(V::Voltage) 
    V = ustrip(Float64, u"mV", V)
    (0.1*(V + 40))/(1 - exp(-(V + 40)/10))
end

function βm(V::Voltage)
    V = ustrip(Float64, u"mV", V)
    4*exp(-(V + 65)/18)
end

function αh(V)
    V = ustrip(Float64, u"mV", V)
    0.07*exp(-(V+65)/20)
end

function βh(V)
    V = ustrip(Float64, u"mV", V)
    1/(1 + exp(-(V + 35)/10))
end

# Steady-state channel gating
m∞(V) = αm(V)/(αm(V) + βm(V))
h∞(V) = αh(V)/(αh(V) + βh(V))
n∞(V) = αn(V)/(αn(V) + βn(V))

@parameters t
params = @parameters Ie gl gna gk gs El Ena Ek Cm
vars = @variables V(t) m(t) h(t) n(t)
D = Differential(t)
# HH equations
eqs = [D(V) ~ (Ie - gl*(V - El) - gna*m^3*h*(V - Ena) - gk*n^4*(V - Ek))/Cm,
       D(m) ~ αm(V) * (1 - m) - βm(V)*m,
       D(h) ~ αh(V) * (1 - h) - βh(V)*h,
       D(n) ~ αn(V) * (1 - n) - βn(V)*n]

sys = ODESystem(eqs, t, [vars...], [])

Vinit = -65.0u"mV"
radius = 0.0025u"cm" # given in cm 
area = 4*pi*(radius)^2 # area in cm²

# Set applied current to zero initially
u0 = [V => Vinit
      m => m∞(Vinit)
      h => h∞(Vinit)
      n => n∞(Vinit)]

p = [Ie => 0.8e-3u"mA" # tonic excitability to force first neuron to fire repetitively
     gl  => 0.3u"mS/cm^2" * area
     gna => 120.0u"mS/cm^2" * area
     gk  => 36.0u"mS/cm^2" * area
     gs  => 0.025u"mS/cm^2" * area
     El  => -54.4u"mV"
     Ena => 50.0u"mV"
     Ek  => -77.0u"mV"
     Cm  => 1.0u"µF/cm^2" * area]

tspan = (0.0u"ms", 250.0u"ms")

prob = ODEProblem(sys, u0, tspan, p)
sol = solve(prob, KenCarp47())
plot(sol, vars=(0,1))
plot!(sol, vars=(0,2))

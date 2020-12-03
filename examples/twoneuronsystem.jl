# This is an example of two synaptically-coupled single-compartment neurons with Hodgkin-Huxley dynamics.
# It's written for illustrative purposes and uses the high-level interface of ModelingToolkit.jl as it already exists today.

using DifferentialEquations, ModelingToolkit, Plots, OrdinaryDiffEq

# Channel gating functions
αn(V) = (0.01*(V + 55))/(1 - exp(-(V + 55)/10))
βn(V) = 0.125 * exp(-(V + 65)/80)
αm(V) = (0.1*(V + 40))/(1 - exp(-(V + 40)/10))
βm(V) = 4*exp(-(V + 65)/18)
αh(V) = 0.07*exp(-(V+65)/20)
βh(V) = 1/(1 + exp(-(V + 35)/10))

# Synaptic transmitter release where
# Tmax = 1 mM Vthreshold = 1 mV Kp = 1 mV
# T(Vpre) = 1 / (1 + exp(-(Vpre - 2)/5))

# Synaptic transmitter release via heaviside step at Vthreshold
heaviside(Vpre) = ifelse(Vpre > 1.0, 1.0, 0.0) 

# Steady-state channel gating
m∞(V) = αm(V)/(αm(V) + βm(V))
h∞(V) = αh(V)/(αh(V) + βh(V))
n∞(V) = αn(V)/(αn(V) + βn(V))

@parameters Ie[1:2] t gl gna gk gs El Ena Ek Es Cm ar ad
@variables V[1:2](t) m[1:2](t) h[1:2](t) n[1:2](t) s[1:2](t)
@derivatives D'~t
@register heaviside(V)
# HH equations
revs = reverse(s)
eqs = [(@. D(V) ~ (Ie - gl*(V - El) - gna*m^3*h*(V - Ena) - gk*n^4*(V - Ek) - gs*revs*(V - Es))/Cm)...,
       (@. D(m) ~ αm(V) * (1 - m) - βm(V)*m)...,
       (@. D(h) ~ αh(V) * (1 - h) - βh(V)*h)...,
       (@. D(n) ~ αn(V) * (1 - n) - βn(V)*n)...,
       (@. D(s) ~ ar*heaviside(V)*(1-s) - ad*s)...]

sys = ODESystem(eqs, t, [V..., m..., h..., n..., s...], [Ie..., gl, gna, gk, gs, El, Ena, Ek, Es, Cm, ar, ad])

Vinit = -65.0
radius = 0.0025 # given in cm 
area = 4*pi*(radius)^2 # area in cm²

# Set applied current to zero initially
u0 = [(V .=> Vinit)...
      (m .=> m∞(Vinit))...
      (h .=> h∞(Vinit))...
      (n .=> n∞(Vinit))...
      (s .=> 0.0)...]

p = [Ie[1] => 0.8e-3 # tonic excitability to force first neuron to fire repetitively
     Ie[2] => 0.0
     gl  => 0.3*area
     gna => 120.0*area
     gk  => 36.0*area
     gs  => 0.025*area
     El  => -54.4
     Ena => 50.0
     Ek  => -77.0
     Es  => 0.0
     Cm  => 1.0e-3*area
     ar  => 1.1
     ad  => 0.19]

tspan = (0.0, 250.0)
prob = ODEProblem(sys, u0, tspan, p)
sol = solve(prob, KenCarp47())
plot(sol, vars=(0,1))
plot!(sol, vars=(0,2))
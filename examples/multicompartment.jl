using ModelingToolkit, DiffEqOperators, OrdinaryDiffEq

# WIP: Attempting to implement a compartmentalized HH cable using
# automatic method of lines discretization via DiffEqOperators. 
# For reference see introduction of: 
# Mascagni 1990 - "The backward Euler method for numerical solution of the Hodgkin-Huxley equations of nerve conductions"

# This may require extension of `discretize()` for a "NeuralSystem"

# Channel gating functions
αn(V) = (0.01*(V + 55))/(1 - exp(-(V + 55)/10))
βn(V) = 0.125 * exp(-(V + 65)/80)
αm(V) = (0.1*(V + 40))/(1 - exp(-(V + 40)/10))
βm(V) = 4*exp(-(V + 65)/18)
αh(V) = 0.07*exp(-(V+65)/20)
βh(V) = 1/(1 + exp(-(V + 35)/10))

# Steady-state channel gating
m∞(V) = αm(V)/(αm(V) + βm(V))
h∞(V) = αh(V)/(αh(V) + βh(V))
n∞(V) = αn(V)/(αn(V) + βn(V))

# Constants
dx = 1.0e-4 # compartment size (1 µm in cm)
Iapp = 5e-9 # in mA
Vinit = -65.0 # mV
a = 2.0e-4 # 2 µm given in cm
r_L = 10000 # 10 kΩ*cm longitudinal resistance
area = 2*pi*a*dx # area of one compartment in cm^2
gl = 0.3*area # mS/cm^2
gna = 120.0*area # ms/cm^2
gk  = 36.0*area # ms/cm^2
El  = -54.4 # mV 
Ena = 50.0 # mV
Ek  = -77.0 # mV
Cm  = 1.0e-3*area # mF / cm^2

@parameters t x 
@variables V(..) m(..) h(..) n(..)
@derivatives Dt'~t
@derivatives Dxx''~x
@derivatives Dx'~x

# HH equations on a cable
eqs = [Dt(V(t,x)) ~ (a/(2*r_L*Cm))*Dxx(V(t,x)) - (gl/Cm)*(V(t,x)-El) - (gna/Cm)*m(t,x)^3*h(t,x)*(V(t,x)-Ena) - (gk/Cm)*n(t,x)^4*(V(t,x)-Ek),
       Dt(m(t,x)) ~ αm(V(t,x)) * (1 - m(t,x)) - βm(V(t,x))*m(t,x),
       Dt(h(t,x)) ~ αh(V(t,x)) * (1 - h(t,x)) - βh(V(t,x))*h(t,x),
       Dt(n(t,x)) ~ αn(V(t,x)) * (1 - n(t,x)) - βn(V(t,x))*n(t,x)]

# Cable equation BCs + ICs
bcs = [V(0.0,x) ~ Vinit,
       Dx(V(t,0.0)) ~ 0.0,
       Dx(V(t,0.1)) ~ 0.0,
       m(0.0,x) ~ m∞(Vinit),
       h(0.0,x) ~ h∞(Vinit),
       n(0.0,x) ~ n∞(Vinit)]

domains = [t ∈ IntervalDomain(0.0, 200.0),
           x ∈ IntervalDomain(0.0, 0.1)]

sys = PDESystem(eqs, bcs, domains, [t, x], [V, m, h, n])

order = 2

discretization = MOLFiniteDifference(dx,order)

prob = discretize(sys,discretization)

sol = solve(prob, KenCarp47())


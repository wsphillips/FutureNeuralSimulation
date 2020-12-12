using ModelingToolkit, DiffEqOperators, OrdinaryDiffEq

# This works after PR in DiffEqOperators for Neumann/Robin BCS gets merged.

# Constants -- these are left out of the parameter list and passed as hard coded
# coefficient values for now.
dx = 1.0e-4 # compartment size (1 µm in cm)
Iapp = 5e-9 # in mA
Vinit = -65.0 # mV
a = 2.0e-4 # 2 µm given in cm
r_L = 10000 # 10 kΩ*cm longitudinal resistance
area = 2*pi*a*dx # area of one compartment in cm^2
gl = 30*area # mS/cm^2 we increase ̅gₗfor illustrative purposes
El  = -54.4 # mV 
Cm  = 1.0e-3*area # mF / cm^2

@parameters t x 
@variables V(..)
@derivatives Dt'~t
@derivatives Dxx''~x
@derivatives Dx'~x

# The Cable Equation with passive leak conductance
eqs = [Dt(V(t,x)) ~ a/(2*r_L*Cm) * Dxx(V(t,x)) - (gl/Cm)*(V(t,x) - El)]

# Cable equation BCs + ICs
bcs = [V(0.0,x) ~ Vinit,
       Dx(V(t,0.0)) ~ -(r_L*Iapp)/(pi*a^2), # Neumann BC for current injection
       Dx(V(t,0.2)) ~ 0.0] # Neumann BC for "sealed end"

domains = [t ∈ IntervalDomain(0.0, 200.0), # simulate 200ms
           x ∈ IntervalDomain(0.0, 0.2)] # over a cable of length 2000 µm

sys = PDESystem(eqs, bcs, domains, [t, x], [V])

order = 2
discretization = MOLFiniteDifference(dx,order)
prob = discretize(sys,discretization)

sol = solve(prob, KenCarp47())


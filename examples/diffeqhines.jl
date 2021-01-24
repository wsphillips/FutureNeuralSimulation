using NeuralSimulation

# Hodgkin-Huxley channel gating functions
αn(V) = V == -55.0 ? 0.1 : (0.01*(V + 55.0))/(1.0 - exp(-(V + 55.0)/10.0))
βn(V) = 0.125 * exp(-(V + 65.0)/80.0)
αm(V) = V == -40.0 ? 1.0 : (0.1*(V + 40.0))/(1.0 - exp(-(V + 40.0)/10.0))
βm(V) = 4.0*exp(-(V + 65.0)/18.0)
αh(V) = 0.07*exp(-(V+65.0)/20.0)
βh(V) = 1.0/(1.0 + exp(-(V + 35.0)/10.0))

# Steady-state channel gating
m∞(V) = αm(V)/(αm(V) + βm(V))
h∞(V) = αh(V)/(αh(V) + βh(V))
n∞(V) = αn(V)/(αn(V) + βn(V))

# Updating gates (rewritten as explicit solution to backwards euler)
@inline function BEgates!(u, p, dt)
    V = view(u, :, 1)
    m = view(u, :, 2)
    h = view(u, :, 3)
    n = view(u, :, 4)
    @inbounds for i in 1:length(V)
        m[i] = (m[i] + αm(V[i])*dt)/(1 + (αm(V[i]) + βm(V[i]))*dt)
        h[i] = (h[i] + αh(V[i])*dt)/(1 + (αh(V[i]) + βh(V[i]))*dt)
        n[i] = (n[i] + αn(V[i])*dt)/(1 + (αn(V[i]) + βn(V[i]))*dt)
    end
end

# In-place update of the Hines coefficients + RHS
@inline function f_hines!(Hines::HinesMatrix{S, BackwardsEuler}, u, p, dt) where {S}
    V = view(u, :, 1)
    m = view(u, :, 2)
    h = view(u, :, 3)
    n = view(u, :, 4)
    a, rl, cm, gna, gk, gl, Ena, Ek, El, Iapp, dx = p
    @inbounds for k in 1:length(V)
        Hines.D[k] = 1 + (a*dt)/(rl*cm*dx^2) + (dt/cm)*(gna*m[k]^3*h[k] + gk*n[k]^4 + gl)
        V[k] +=  (dt/cm)*(gna*m[k]^3*h[k]*Ena + gk*n[k]^4*Ek + gl*El)
    end
    # current injected to first compartment; for testing only!
    V[1] += (Iapp*dt)/(pi*a*cm*dx)
    return nothing
end

##############################################################

dt = 0.025 # ms
Iapp = 1.0e-6 # mA (1 nA)
cable_length = 2000.0e-3 # 2000 µm length
N = 2000
dx = cable_length/N
radius = 2.0e-3 # µm in mm
area = 2*pi*radius*dx

rl = 1.0e6 # 1 kΩ in milliohms*mm
gl  = 0.003 #mS/mm^2
gna = 1.2 # mS/mm^2 (modified from 1.2 mS)
gk  = .36 # mS/mm^2
El  = -54.4 # mV
Ena = 50.0 # mV
Ek  = -77.0 # mV
cm  = 1.0e-5 # mF/mm^2

Vinit = -55.0 # mV
tspan = (0.0, 200.0)

p = [radius, rl, cm, gna, gk, gl, Ena, Ek, El, Iapp, dx]

cable = CableSection(radius, cable_length, rl, cm, N)

u0 = zeros(2000,4)
u0[:,1] .= Vinit
u0[:,2] .= m∞(Vinit)
u0[:,3] .= h∞(Vinit)
u0[:,4] .= n∞(Vinit)

fn = HHCableFunction{true}(f_hines!, BEgates!)
prob = HHCableProblem(fn, cable, u0, tspan, p) 
@time sol = solve(prob, BackwardsEuler(), dt = 0.025)


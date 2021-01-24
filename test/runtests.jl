using Test
using NeuralSimulation

# Hodgkin-Huxley channel gating functions
αn(V) = V == -55.0 ? 0.1 : (0.01*(V + 55))/(1 - exp(-(V + 55)/10))
βn(V) = 0.125 * exp(-(V + 65)/80)
αm(V) = V == -40.0 ? 1.0 : (0.1*(V + 40))/(1 - exp(-(V + 40)/10))
βm(V) = 4*exp(-(V + 65)/18)
αh(V) = 0.07*exp(-(V+65)/20)
βh(V) = 1/(1 + exp(-(V + 35)/10))

# Steady-state channel gating
m∞(V) = αm(V)/(αm(V) + βm(V))
h∞(V) = αh(V)/(αh(V) + βh(V))
n∞(V) = αn(V)/(αn(V) + βn(V))

# Updating gates (generic version)
function gates!(du, u, p, t)
    V, m, h, n = [view(u, :, x) for x in 1:4]
    _, dm, dh, dn = [view(du, :, x) for x in 1:4]

    for i in 1:length(V)
        dm[i] = αm(V[i]) * (1 - m[i]) - βm(V[i])*m[i]
        dh[i] = αh(V[i]) * (1 - h[i]) - βh(V[i])*h[i]
        dn[i] = αn(V[i]) * (1 - n[i]) - βn(V[i])*n[i]
    end
end

# Updating gates (rewritten for backwards euler)
function BEgates!(u, p, dt)
    V, m, h, n = [view(u, :, x) for x in 1:4]

    for i in 1:length(V)
        m[i] = (m[i] + αm(V[i])*dt)/(1 + (αm(V[i]) + βm(V[i]))*dt)
        h[i] = (h[i] + αh(V[i])*dt)/(1 + (αh(V[i]) + βh(V[i]))*dt)
        n[i] = (n[i] + αn(V[i])*dt)/(1 + (αn(V[i]) + βn(V[i]))*dt)
    end
end

# TODO: Hines matrix prototype -> should use ModelingToolkit IR to build a function from
# a `ModelingToolkit.Equation` If we pass a keyword, like `HinesBackwardsEuler=true` it
# should generate a function that spits out updated coefficients for the tridiagonal solve
# of the cable equation 
mutable struct HinesMatrix{S}
    L::S
    D::S
    U::S
end

# TODO: other constructors for more elaborate tree structure types
# this version is specific to backwards Euler
function HinesMatrix(cable::CableSection, dt)
        rl = cable.rl
        a = cable.radius
        cm = cable.cm
        n = cable.num
        dx = cable.len/n

        L = Vector{typeof(cable.len)}(undef, n)
        D = similar(L)
        U = similar(L)

        # L and U are static
        L[1] = 0.0
        L[2:end-1] .= -(a*dt)/(2*rl*cm*dx^2)
        L[end] = -(a*dt)/(rl*cm*dx^2)
        
        U[1] = -(a*dt)/(rl*cm*dx^2)
        U[2:end-1] .= -(a*dt)/(2*rl*cm*dx^2)
        U[end] = 0.0

        return HinesMatrix(L, D, U)
end

# In-place update of the Hines coefficients + RHS (`u` -> Vm)
function f_hines!(Hines, u, p, dt)
    V, m, h, n = [view(u, :, x) for x in 1:4]
    a, rl, cm, gna, gk, gl, Ena, Ek, El, Iapp, dx = p
    for k in 1:length(V)
        Hines.D[k] = 1 + (a*dt)/(rl*cm*dx^2) + (dt/cm)*(gna*m[k]^3*h[k] + gk*n[k]^4 + gl)
        V[k] += (dt/cm)*(gna*m[k]^3*h[k]*Ena + gk*n[k]^4*Ek + gl*El)
    end 
    # in first compartment we inject additional 1 nA current
    # (for testing only; stimulation should be specified at high level)
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
area = 2*pi*a*dx

rl = 1.0e6 # 1 kΩ in milliohms*mm
gl  = 0.003 #mS/mm^2
gna = 1.2 # mS/mm^2 (modified from 1.2 mS)
gk  = .36 # mS/mm^2
El  = -54.4 # mV
Ena = 50.0 # mV
Ek  = -77.0 # mV
cm  = 1.0e-5 # mF/mm^2

Vinit = -55.0 # mV

p = [radius, rl, cm, gna, gk, gl, Ena, Ek, El, Iapp, dx]

foo = CableSection(radius, cable_length, rl, cm, N)
Hmat = HinesMatrix(foo, 0.025)

u0 = zeros(2000,4)

u0[:,1] .= Vinit
u0[:,2] .= m∞(Vinit)
u0[:,3] .= h∞(Vinit)
u0[:,4] .= n∞(Vinit)

fn = HHCableFunction(f_hines!, BEgates!)
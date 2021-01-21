using Test
using NeuronalSimulation

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

function stepgates!(u, p, dt)
    V, m, h, n = [view(u, :, x) for x in 1:4]
    for i in 1:length(V)
        m[i] = (m[i] + αm(V[i])*dt)/(1 + (αm(V[i]) + βm(V[i]))*dt)
        h[i] = (h[i] + αh(V[i])*dt)/(1 + (αh(V[i]) + βh(V[i]))*dt)
        n[i] = (n[i] + αn(V[i])*dt)/(1 + (αn(V[i]) + βn(V[i]))*dt)
    end
end

# Eventually we want to use ModelingToolkit IR to build a function from a `ModelingToolkit.Equation`
# If we pass a keyword, like `HinesBackwardsEuler=true` it should generate a function that spits out
# updated coefficients for the tridiagonal solve of the cable equation 

mutable struct HinesMatrix
    L::S
    D::S
    U::S
end

# Eventually other constructors for more elaborate tree structure types
function HinesMatrix(cable::CableSection, dt)
        rl = cable.rl
        a = cable.radius
        cm = cable.cm
        n = cable.num
        dx = cable.len/n

        L = Vector{typeof(cable.len)}(undef, n)
        D = copy(L)
        U = copy(L)

        # L and U are static
        L[1] = 0.0
        L[2:end-1] .= -(a*dt)/(2*rl*cm*dx^2)
        L[end] = -(a*dt)/(rl*cm*dx^2)
        
        U[1] = -(a*dt)/(rl*cm*dx^2)
        U[2:end-1] .= -(a*dt)/(2*rl*cm*dx^2)
        U[end] = 0.0

        new(L, D, U)
end

# In-place update of the Hines coefficients + RHS (`u` -> Vm)
function f_hines!(Hines, u, p, dt)
    V, m, h, n = [view(u, :, x) for x in 1:4]
    
    # FIXME: these parameters need to be setup correctly
    a, rl, cm, gna, gk, gl, Ena, Ek, El, Iapp, dx = p

    for k in 1:length(V)
        Hines.D[k] = 1 + (a*dt)/(rl*cm*dx^2) + (dt/cm)*(gna*m[k]^3*h[k] + gk*n[k]^4 + gl)
        V[k] = V[k] + (dt/cm)*(gna*m[k]^3*h[k]*Ena + gk*n[k]^4*Ek + gl*El)
    end 

    # in first compartment we inject additional 1 nA current
    # (for testing only; stimulation should be specified at high level)
    V[1] += (Iapp*dt)/(pi*a*cm*dx)
    
    #Backward pass
    for i in N:-1:2
        factor = Hines.U[i] / Hines.D[i]
        Hines.D[i-1] -= factor * Hines.L[i] 
        V[i-1] -= factor * V[i] 
    end

    # Root solve
    V[1] /= Hines.D[1]

    # Forward pass
    for j in 2:N
        V[j] -= Hines.L[j]*V[j-1]
        V[j] /= Hines.D[j]
    end
end



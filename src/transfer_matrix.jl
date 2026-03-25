
using StaticArrays, LinearAlgebra

export Dist, Pos, transfer_matrix

const c0 = 299792458.

# abstract utility types to differentiate between position space and distance space input
abstract type Space end
abstract type Dist <: Space end
abstract type Pos  <: Space end


# transfer matrix algorithm for distance space input
function transfer_matrix(::Type{Dist},freqs::Union{Real,AbstractVector{<:Real}},
        distances::AbstractVector{<:Real};
        eps::Real=24.0,tand::Real=0.0,thickness::Real=1e-3,nm::Real=1e15)

    ϵ  = eps*(1.0-1.0im*tand)
    nd = sqrt(ϵ); nm = complex(nm)
    ϵm = nm^2
    A  = 1-1/ϵ
    A0 = 1-1/ϵm

    R = Vector{ComplexF64}(undef,length(freqs))
    B = Vector{Float64}(undef,length(freqs))

    Gd = SMatrix{2,2,ComplexF64}((1+nd)/2,   (1-nd)/2,   (1-nd)/2,   (1+nd)/2)
    Gv = SMatrix{2,2,ComplexF64}((nd+1)/2nd, (nd-1)/2nd, (nd-1)/2nd, (nd+1)/2nd)
    G0 = SMatrix{2,2,ComplexF64}((1+nm)/2,   (1-nm)/2,   (1-nm)/2,   (1+nm)/2)
    T  = MMatrix{2,2,ComplexF64}(undef); copyto!(T,Gd)

    S  = SMatrix{2,2,ComplexF64}( A/2, 0.0im, 0.0im,  A/2)
    S0 = SMatrix{2,2,ComplexF64}(A0/2, 0.0im, 0.0im, A0/2)
    M  = MMatrix{2,2,ComplexF64}(undef); copyto!(M,S)

    W  = MMatrix{2,2,ComplexF64}(undef) # work matrix for multiplication

    @inbounds @views for j in eachindex(freqs)
        pd1 = cispi(-2*freqs[j]*nd*thickness/c0)
        pd2 = cispi(+2*freqs[j]*nd*thickness/c0)
        
        # iterate in reverse order to sum up M in single sweep (thx david)
        for i in Iterators.reverse(eachindex(distances))
            T[:,1] .*= pd1
            T[:,2] .*= pd2 # T = Gd*Pd

            # mul!(W,T,S); M .-= W        # M = Gd*Pd*S_-1
            mul!(M,T,S,-1.,1.)
            mul!(W,T,Gv); copyto!(T,W)  # T *= Gd*Pd*Gv

            T[:,1] .*= cispi(-2*freqs[j]*distances[i]/c0)
            T[:,2] .*= cispi(+2*freqs[j]*distances[i]/c0)   # T = Gd*Pd*Gv*Gd*S_-1

            if i > 1
                mul!(M,T,S,1.,1.)
                mul!(W,T,Gd); copyto!(T,W)
            else
                mul!(M,T,S0,1.,1.)
                mul!(W,T,G0); copyto!(T,W)
            end
        end
        
        R[j] = T[1,2]/T[2,2]
        B[j] = abs2(M[1,1]+M[1,2]-(M[2,1]+M[2,2])*T[1,2]/T[2,2])

        copyto!(M,S)
        T .= 1.0+0.0im; T[1,1] += nd; T[2,2] += nd; T[2,1] -= nd; T[1,2] -= nd; T .*= 0.5
    end

    return R, B
end

transfer_matrix(freqs::Union{Real,AbstractVector{<:Real}},distances::AbstractVector{<:Real};
    eps::Real=24.0,tand::Real=0.0,thickness::Real=1e-3,nm::Real=1e15) =
    transfer_matrix(Dist,freqs,distances; tand=tand,eps=eps,thickness=thickness,nm=nm)



# transfer matrix algorithm for position space input
function transfer_matrix(::Type{Pos},freqs::Union{Real,AbstractVector{<:Real}},
        position::AbstractVector{<:Real};
        eps::Real=24.0,tand::Real=0.0,thickness::Real=1e-3,nm::Real=1e15)

    ϵ  = eps*(1.0-1.0im*tand)
    nd = sqrt(ϵ); nm = complex(nm)
    ϵm = nm^2
    A  = 1-1/ϵ
    A0 = 1-1/ϵm

    R = Vector{ComplexF64}(undef,length(freqs))
    B = Vector{Float64}(undef,length(freqs))

    Gd = SMatrix{2,2,ComplexF64}((1+nd)/2,   (1-nd)/2,   (1-nd)/2,   (1+nd)/2)
    Gv = SMatrix{2,2,ComplexF64}((nd+1)/2nd, (nd-1)/2nd, (nd-1)/2nd, (nd+1)/2nd)
    G0 = SMatrix{2,2,ComplexF64}((1+nm)/2,   (1-nm)/2,   (1-nm)/2,   (1+nm)/2)
    T  = MMatrix{2,2,ComplexF64}(undef); copyto!(T,Gd)

    S  = SMatrix{2,2,ComplexF64}( A/2, 0.0im, 0.0im,  A/2)
    S0 = SMatrix{2,2,ComplexF64}(A0/2, 0.0im, 0.0im, A0/2)
    M  = MMatrix{2,2,ComplexF64}(undef); copyto!(M,S)

    W  = MMatrix{2,2,ComplexF64}(undef) # work matrix for multiplication

    @inbounds @views for j in eachindex(freqs)
        pd1 = cispi(-2*freqs[j]*nd*thickness/c0)
        pd2 = cispi(+2*freqs[j]*nd*thickness/c0)

        # iterate in reverse order to sum up M in single sweep (thx david)
        for i in Iterators.reverse(eachindex(position))
            T[:,1] .*= pd1
            T[:,2] .*= pd2 # T = Gd*Pd

            # mul!(W,T,S); M .-= W        # M = Gd*Pd*S_-1
            mul!(M,T,S,-1.,1.)
            mul!(W,T,Gv); copyto!(T,W)  # T *= Gd*Pd*Gv

            d = position[i]-(i==1 ? 0 : position[i-1]+thickness)
            T[:,1] .*= cispi(-2*freqs[j]*d/c0)
            T[:,2] .*= cispi(+2*freqs[j]*d/c0)   # T = Gd*Pd*Gv*Gd*S_-1

            if i > 1
                # mul!(W,T,S); M .+= W
                mul!(M,T,S,1.,1.)
                mul!(W,T,Gd); copyto!(T,W)
            else
                # mul!(W,T,S0); M .+= W
                mul!(M,T,S0,1.,1.)
                mul!(W,T,G0); copyto!(T,W)
            end
        end
        
        R[j] = T[1,2]/T[2,2]
        B[j] = abs2(M[1,1]+M[1,2]-(M[2,1]+M[2,2])*T[1,2]/T[2,2])

        copyto!(M,S)
        # T .= 1.0+0.0im; T[1,1] += nd; T[2,2] += nd; T[2,1] -= nd; T[1,2] -= nd; T .*= 0.5
        copyto!(T,Gd)
    end

    return R, B
end



@doc """
transfer_matrix(::Type{Space},freqs::Union{Real,AbstractVector{<:Real}},
    disc_configuration::AbstractVector{<:Real};
    eps::Real=24.0,tand::Real=0.0,thickness::Real=1e-3,nm::Real=1e15)::Matrix{ComplexF64}

Return vectors containing (complex) reflectivity and boost of the
[transfer matrix algorithm](https://arxiv.org/pdf/1612.07057) for a disc system.

Use `Dist, Pos <: Space` to denote whether `disc_configuration` is given in distance space or position space.
Omitting `Space` defaults to distance space.

#Examples
```jldoctest
julia> ref, boost = transfer_matrix(Dist,22e9:0.01e9:22.05e9,[1,2,3,4]*1e-3)
(ComplexF64[
 -0.9208011629309648 + 0.3900323298714899im 
 -0.9203949732331749 + 0.3909898889320582im 
 -0.9199866293633413 + 0.39194974396302146im
 -0.9195761162569096 + 0.392911906678658im
 -0.9191634187090357 + 0.39387638886726645im
 -0.9187485213729765 + 0.39484320239174653im],
 Float64[
 0.34105872984599983
 0.339972315462861
 0.33888304718993706
 0.33779091882570444
 0.3366959242101794
 0.3355980572261133],
)
```
""" transfer_matrix





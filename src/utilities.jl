
export pos2dist, dist2pos, gen_freqs

"""
    pos2dist(position::Array{<:Real}; thickness::Real=1e-3)

Return distances corresponding to `position`.
"""
function pos2dist(position::Array{<:Real}; thickness::Real=1e-3)
    pos = [0; position]
    d = (pos[2:end]-pos[1:end-1])
    d[2:end] .-= thickness
    
    return d
end

"""
    dist2pos(distances::Array{<:Real}; thickness::Real=1e-3)

Return position corresponding to `distances`.
"""
function dist2pos(distances::Array{<:Real}; thickness::Real=1e-3)
    return [sum(distances[1:i])+(i-1)*thickness for i in 1:length(distances)]
end



"""
    gen_freqs(fcenter::Real,fwidth::Real; n::Int=100)

Return `n` equally spaced frequencies from `fcenter-fwidth/2` to
`fcenter+fwidth/2`.
"""
function gen_freqs(fcenter::Real,fwidth::Real; n::Int=100)
    return collect(range(fcenter-fwidth/2; stop=fcenter+fwidth/2,length=n))
end

"""
    gen_freqs(bounds; n::Int=100)

Return `n` equally spaced frequencies from `bounds[1]` to `bounds[2]`.
"""
function gen_freqs(bounds::Union{Tuple,AbstractVector{<:Real}}; n::Int=100)
    @assert length(bounds) == 2 "Bounds needs to be a collection of 2 values."

    return collect(range(bounds[1],bounds[2],n))
end

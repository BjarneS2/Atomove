module Basis3D

export build_s_basis, build_ua_basis

function build_s_basis(n::Int, N_modes::Int)
    τ = range(0.0, 1.0, length=n)
    win = sin.(π .* τ) .^ 2
    Φ = zeros(n, 2 * N_modes)
    for k in 1:N_modes
        Φ[:, 2k-1] = win .* cos.(2π * k .* τ)
        Φ[:, 2k] = win .* sin.(2π * k .* τ)
    end
    return Φ
end

function build_ua_basis(n::Int, N_modes::Int)
    τ = range(0.0, 1.0, length=n)
    win = sin.(π .* τ)
    Φ = zeros(n, 2 * N_modes)
    for k in 1:N_modes
        Φ[:, 2k-1] = win .* cos.(2π * k .* τ)
        Φ[:, 2k] = win .* sin.(2π * k .* τ)
    end
    return Φ
end

end

module Types3D

export TweezerParams3D, ControlProtocol3D, Trajectory3D, InitialG3D
export ControlBounds3D, default_bounds3d
export PhysicalConstants3D, default_constants3d
export ThermalControlResult3D
export transport_direction, transport_length

Base.@kwdef struct PhysicalConstants3D
    kB::Float64 = 1.380649e-23
    hbar::Float64 = 1.054571817e-34
    m::Float64 = 2.20695e-25
    g_SI::Float64 = 9.81
    w0_um::Float64 = 1.0
    t0_us::Float64 = 1.0
    wavelength_static_nm::Float64 = 1064.0
    wavelength_dynamic_nm::Float64 = 934.0
end

default_constants3d() = PhysicalConstants3D()

function compute_scales3d(consts::PhysicalConstants3D)
    w0_SI = consts.w0_um * 1e-6
    t0_SI = consts.t0_us * 1e-6
    v0 = w0_SI / t0_SI
    E0 = consts.m * v0^2
    g_dimless = consts.g_SI * t0_SI^2 / w0_SI
    return (w0_SI=w0_SI, t0_SI=t0_SI, v0=v0, E0=E0, g_dimless=g_dimless)
end

Base.@kwdef struct TweezerParams3D
    w::Float64
    w_aux_factor::Float64 = 1.0
    zR::Float64
    zR_aux::Float64
    x_start::Float64 = 0.0
    y_start::Float64 = 0.0
    x_stop::Float64
    y_stop::Float64 = 0.0
    n::Int
    maxT::Float64
    U0_static::Float64
    U0_aux_max::Float64
    T_atom::Float64 = 40e-6
    T_tweezer::Float64 = 287e-6
    starting_trap_fraction::Float64 = 0.5
    trap_fraction::Union{Float64,Nothing} = 0.5
    final_trap_fraction::Float64 = 0.5
    lambda_heat::Float64 = 0.0
    lambda_jitter_pos::Float64 = 0.0
    lambda_jitter_ua::Float64 = 0.0
    max_sigma_position::Float64 = 2.0
    move_in_single_trap::Bool = false
    single_trap_amplitude::Float64 = 1.0
    z_aux_offset::Float64 = 0.0
end

function transport_direction(p::TweezerParams3D)
    dx = p.x_stop - p.x_start
    dy = p.y_stop - p.y_start
    L = sqrt(dx^2 + dy^2)
    return dx/L, dy/L, L
end

transport_length(p::TweezerParams3D) = sqrt((p.x_stop - p.x_start)^2 + (p.y_stop - p.y_start)^2)

Base.@kwdef struct ControlBounds3D
    T_min_fraction::Float64 = 0.05
    r_margin_w::Float64 = 0.5
    z_margin::Float64 = 2.0
    v_xy_max::Float64 = 2.0
    v_z_max::Float64 = 2.0
    u_margin_w::Float64 = 1.0
    ua_min::Float64 = 0.0
    ua_max::Float64 = 1.0
    v_u_max_per_w::Float64 = 3.0
    v_ua_max::Float64 = 5.0
end

default_bounds3d(::TweezerParams3D) = ControlBounds3D()

struct ControlProtocol3D
    t::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    vx::Vector{Float64}
    vy::Vector{Float64}
    vz::Vector{Float64}
    ux::Vector{Float64}
    uy::Vector{Float64}
    ua::Vector{Float64}
end

struct Trajectory3D
    t::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    vx::Vector{Float64}
    vy::Vector{Float64}
    vz::Vector{Float64}
    lost::Bool
end

struct InitialG3D
    dt::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    vx::Vector{Float64}
    vy::Vector{Float64}
    vz::Vector{Float64}
    ux::Vector{Float64}
    uy::Vector{Float64}
    ua::Vector{Float64}
end

struct ThermalControlResult3D
    protocol::ControlProtocol3D
    samples::Vector{NamedTuple}
    x_traj::Matrix{Float64}
    y_traj::Matrix{Float64}
    z_traj::Matrix{Float64}
    vx_traj::Matrix{Float64}
    vy_traj::Matrix{Float64}
    vz_traj::Matrix{Float64}
    seed::Union{Nothing,Int}
    n_samples::Int
    termination_status::String
    objective_value::Float64
end

end

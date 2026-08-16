module ThermalSampling3D

using ..Types3D
using ..Models3D: potential3d
using Random

export compute_scales3d_full, sample_initial_conditions3d, is_trapped3d

# Returns the full scale namedtuple including g_dimless.
function compute_scales3d_full(p::TweezerParams3D;
                                consts::PhysicalConstants3D = default_constants3d())
    w0_SI  = consts.w0_um * 1e-6
    t0_SI  = consts.t0_us * 1e-6
    v0     = w0_SI / t0_SI
    E0     = consts.m * v0^2

    if p.move_in_single_trap
        # Trap depth at t=0 in SI: single_trap_amplitude * kB * T_tweezer
        U0_SI  = p.single_trap_amplitude * consts.kB * p.T_tweezer
        w_SI   = p.w * p.w_aux_factor * w0_SI
        zR_SI  = p.zR_aux * w0_SI
    else
        U0_SI  = consts.kB * p.T_tweezer
        w_SI   = p.w * w0_SI   # static trap waist (p.w is in units of w0)
        zR_SI  = p.zR * w0_SI
    end

    # radial trap frequency from harmonic approximation: ωr² = 4U0/(m w²)
    omega_r = sqrt(4.0 * U0_SI / (consts.m * w_SI^2))
    sigma_r = sqrt(consts.kB * p.T_atom / (consts.m * omega_r^2))   # [m]
    sigma_r_dimless = sigma_r / w0_SI

    # axial trap frequency: ωz² = 2U0/(m zR²)
    omega_z = sqrt(2.0 * U0_SI / (consts.m * zR_SI^2))
    sigma_z = sqrt(consts.kB * p.T_atom / (consts.m * omega_z^2))   # [m]
    sigma_z_dimless = sigma_z / w0_SI

    sigma_v = sqrt(consts.kB * p.T_atom / consts.m)   # [m/s]
    sigma_v_dimless = sigma_v / v0

    g_dimless = consts.g_SI * t0_SI^2 / w0_SI

    return (
        w0_SI           = w0_SI,
        t0_SI           = t0_SI,
        v0              = v0,
        E0              = E0,
        sigma_r_dimless = sigma_r_dimless,
        sigma_z_dimless = sigma_z_dimless,
        sigma_v_dimless = sigma_v_dimless,
        omega_r         = omega_r,
        omega_z         = omega_z,
        g_dimless       = g_dimless,
    )
end

function is_trapped3d(x::Float64, y::Float64, z::Float64,
                       vx::Float64, vy::Float64, vz::Float64,
                       ux::Float64, uy::Float64,
                       p::TweezerParams3D)
    if p.move_in_single_trap
        w_aux = p.w * p.w_aux_factor
        zR_a  = p.zR_aux
        rho2  = (x - ux)^2 + (y - uy)^2
        wXi2  = w_aux^2 * (1.0 + (z / zR_a)^2)
        fa    = (w_aux^2 / wXi2) * exp(-2.0 * rho2 / wXi2)
        U     = -p.single_trap_amplitude * p.U0_static * fa
    else
        U = potential3d(x, y, z, ux, uy, 0.0, p)
    end
    KE = 0.5 * (vx^2 + vy^2 + vz^2)
    return (U + KE) < p.starting_trap_fraction * U
end

function sample_initial_conditions3d(
    p::TweezerParams3D;
    consts::PhysicalConstants3D = default_constants3d(),
    check_trapped::Bool   = true,
    max_attempts::Int     = 2000,
)
    scales = compute_scales3d_full(p; consts = consts)

    sr = scales.sigma_r_dimless
    sz = scales.sigma_z_dimless
    sv = scales.sigma_v_dimless

    attempts = 0
    while attempts < max_attempts
        attempts += 1

        dx = randn() * sr
        dy = randn() * sr
        dz = randn() * sz
        vx = randn() * sv
        vy = randn() * sv
        vz = randn() * sv

        x = p.x_start + dx
        y = 0.0       + dy
        z = 0.0       + dz

        if !check_trapped || is_trapped3d(x, y, z, vx, vy, vz, p.x_start, 0.0, p)
            return (
                x = x, y = y, z = z,
                vx = vx, vy = vy, vz = vz,
                scales = scales,
            )
        end
    end
    error("Failed to sample trapped atom after $max_attempts attempts.")
end

end

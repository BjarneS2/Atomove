module ForwardDynamics3D

using ..Types3D
using ..Models3D
using ..ThermalSampling3D

export simulate_forward3d

function simulate_forward3d(
    ctrl::ControlProtocol3D,
    params::TweezerParams3D;
    thermal_sample::Bool=true,
    consts::PhysicalConstants3D=default_constants3d(),
    return_info::Bool=false,
    init::Union{Nothing,NamedTuple}=nothing,
)
    scales = compute_scales3d_full(params; consts=consts)
    g = scales.g_dimless

    if init === nothing
        init = sample_initial_conditions3d(
            params;
            consts=consts,
            check_trapped=thermal_sample,
        )
    end

    t = ctrl.t
    ux = ctrl.ux
    uy = ctrl.uy
    ua = ctrl.ua
    n = length(t)

    x = zeros(Float64, n);
    x[1] = init.x
    y = zeros(Float64, n);
    y[1] = init.y
    z = zeros(Float64, n);
    z[1] = init.z
    vx = zeros(Float64, n);
    vx[1] = init.vx
    vy = zeros(Float64, n);
    vy[1] = init.vy
    vz = zeros(Float64, n);
    vz[1] = init.vz
    lost = false

    for j in 1:(n-1)
        dt = t[j+1] - t[j]
        Fx0, Fy0, Fz0 = forces3d(x[j], y[j], z[j], ux[j], uy[j], ua[j], params, g)
        x[j+1] = x[j] + vx[j] * dt + 0.5 * Fx0 * dt^2
        y[j+1] = y[j] + vy[j] * dt + 0.5 * Fy0 * dt^2
        z[j+1] = z[j] + vz[j] * dt + 0.5 * Fz0 * dt^2
        Fx1, Fy1, Fz1 = forces3d(x[j+1], y[j+1], z[j+1], ux[j+1], uy[j+1], ua[j+1], params, g)
        vx[j+1] = vx[j] + 0.5 * (Fx0 + Fx1) * dt
        vy[j+1] = vy[j] + 0.5 * (Fy0 + Fy1) * dt
        vz[j+1] = vz[j] + 0.5 * (Fz0 + Fz1) * dt
    end

    # Loss check: total energy > trap_fraction * potential at each step
    # (trap_fraction === nothing disables this cutoff, e.g. for stress testing)
    if params.trap_fraction !== nothing
        for j in 1:n
            U = potential3d(x[j], y[j], z[j], ux[j], uy[j], ua[j], params)
            KE = 0.5 * (vx[j]^2 + vy[j]^2 + vz[j]^2)
            if (U + KE + g * y[j]) > params.trap_fraction * U
                lost = true
                break
            end
        end
    end

    traj = Trajectory3D(t, x, y, z, vx, vy, vz, lost)

    if return_info
        w0_SI = scales.w0_SI
        t0_SI = scales.t0_SI
        v0 = scales.v0
        physical = (
            t=t .* t0_SI,
            x=x .* w0_SI,
            y=y .* w0_SI,
            z=z .* w0_SI,
            vx=vx .* v0,
            vy=vy .* v0,
            vz=vz .* v0,
        )
        return traj, (; physical=physical, scales=scales, init=init)
    end

    return traj
end

end

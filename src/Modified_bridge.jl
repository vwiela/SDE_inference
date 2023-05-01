"""
    calc_log_det(X::T1, dim_X::T2) where {T1<:MArray{<:Tuple, Float64, 2}, T2<:Signed}

Calculate log-determinant for a semite positive definite matrix X of dimension dim_X. 
"""
@inline function calc_log_det(X::T1, dim_X::T2) where {T1<:MArray{<:Tuple, Float64, 2}, T2<:Signed}
    log_det::Float64 = 0.0
    @inbounds @simd for i in 1:dim_X
        log_det += log(X[i, i])
    end
    log_det *= 2
    return log_det
end


"""
    calc_log_pdf_mvn(x_curr::T1, 
                     mean_vec::T1, 
                     chol_cov_mat::T2, 
                     dim_mod::T3)::Float64 where{T1<:MArray{<:Tuple, Float64, 1}, T2<:MArray{<:Tuple, Float64, 2}, T3<:Signed}

Calculate log-pdf for a multivariate normal distribution at x_curr. 

Since already computed the cholesky decomposition of the covariance matrix is employed. 
"""
function calc_log_pdf_mvn(x_curr::T1, 
                          mean_vec::T1, 
                          chol_cov_mat::T2, 
                          dim_mod::T3)::Float64 where{T1<:MArray{<:Tuple, Float64, 1}, T2<:MArray{<:Tuple, Float64, 2}, T3<:Signed}

    # log(2pi)
    const_term::Float64 = 1.837877066

    # (x - mu)' * inv_∑ * (x - mu)
    MD2 = chol_cov_mat \ (x_curr - mean_vec)
    MD2 = dot(MD2, MD2)

    log_pdf::Float64 = -0.5 * (const_term*dim_mod + calc_log_det(chol_cov_mat, dim_mod) + MD2)

    return log_pdf
end


"""
    init_sol_object(::Val{dim_model}, ::Val{dim_model_obs}, sde_mod::SdeModel)::DiffBridgeSolverObj where {dim_model, dim_model_obs}

Initialise solution-struct (DiffBridgeSolverObj) to pre-allocate matrices and vectors for propgating particles. 

Pre-allocates drift-vector (for both EM and bridge), diffusion-matrix (both EM and bridge), current particle-values at time, 
and step-length to propegate the particles in a memory efficient manner. As StaticArrays-are employed Val-input required to 
help compiler. 
"""
function init_sol_object(::Val{dim_model}, ::Val{dim_model_obs}, sde_mod::SdeModel)::DiffBridgeSolverObj where {dim_model, dim_model_obs}
    
    mean_vec = zeros(MVector{dim_model, Float64})
    alpha_vec = zeros(MVector{dim_model, Float64})
    
    beta_mat = zeros(MMatrix{dim_model, dim_model, Float64})
    cov_mat = zeros(MMatrix{dim_model, dim_model, Float64})

    x_curr = zeros(MVector{dim_model, Float64})

    sigma_mat = zeros(MMatrix{dim_model_obs, dim_model_obs, Float64})

    P_mat = deepcopy(sde_mod.P_mat)
    P_mat_t = deepcopy(sde_mod.P_mat')

    Δt = Array{Float64, 1}(undef, 1)
    sqrt_Δt = Array{Float64, 1}(undef, 1)

    solver_obj = DiffBridgeSolverObj(mean_vec,
                                     alpha_vec,
                                     cov_mat,
                                     beta_mat,
                                     sigma_mat,
                                     P_mat,
                                     P_mat_t,
                                     x_curr,
                                     Δt,
                                     sqrt_Δt)

    return solver_obj
end


"""
    modified_diffusion_calc_arrays!(p::DynModInput,                                           
                                    sde_mod::SdeModel, 
                                    so::DiffBridgeSolverObj, 
                                    delta_k::Float64,
                                    y_vec,
                                    t::Float64) 

Calculate arrays (drift- and diffusion for both EM and bridge) to propegate via modified diffusion bridge 
"""
function modified_diffusion_calc_arrays!(p::DynModInput,                                           
                                         sde_mod::SdeModel, 
                                         so::DiffBridgeSolverObj, 
                                         delta_k::Float64,
                                         y_vec,
                                         t::Float64) 
    
    # Calculate beta and alpha arrays 
    sde_mod.calc_alpha(so.alpha_vec, so.x_vec, p, t)
    sde_mod.calc_beta(so.beta_mat, so.x_vec, p, t)

    # Structs for calculating mean and covariance
    Δt::Float64 = so.Δt[1]
    sqrt_Δt::Float64 = so.sqrt_Δt[1]

    # Calculate new mean and covariance values 
    inv_term = so.beta_mat*so.P_mat*inv(so.P_mat_t*so.beta_mat*so.P_mat*delta_k + so.sigma_mat)

    so.mean_vec .= so.alpha_vec .+ inv_term * (y_vec - so.P_mat_t * (so.x_vec + so.alpha_vec * delta_k))
    
    so.cov_mat .= so.beta_mat .- inv_term*so.P_mat_t*so.beta_mat*Δt

end


"""
    modified_diffusion_propegate!(prob_em::Array{Float64, 1}, 
                                  prob_bridge::Array{Float64, 1}, 
                                  so::DiffBridgeSolverObj, 
                                  u_vec, 
                                  i_particle::T1, 
                                  sde_mod::SdeModel) where T1<:Signed

Propegate the particles one time-step for the modified diffusion bridge. 
"""
function modified_diffusion_propegate!(prob_em::Array{Float64, 1}, 
                                       prob_bridge::Array{Float64, 1}, 
                                       so::DiffBridgeSolverObj, 
                                       u_vec, 
                                       i_particle::T1, 
                                       sde_mod::SdeModel) where T1<:Signed
     
    Δt::Float64 = so.Δt[1]
    sqrt_Δt::Float64 = so.sqrt_Δt[1]

    # Must calculate mean-vectors before propegating (when calculcating logpdf)
    mean_vec_bridge = so.x_vec + so.mean_vec*Δt
    mean_vec_em = so.x_vec + so.alpha_vec*Δt

    # Note, for propegation and then log-pdf cholesky decompositon is required for cov-mat and beta 
    calc_cholesky!(so.cov_mat, sde_mod.dim)
    calc_cholesky!(so.beta_mat, sde_mod.dim)

    # Propegate
    so.x_vec .+= so.mean_vec*Δt + so.cov_mat*u_vec * sqrt_Δt
    map_to_zero!(so.x_vec, sde_mod.dim)
    
    # Update probabilities, cov_mat and beta-mat are both lower-triangular cholesky
    prob_bridge[i_particle] += calc_log_pdf_mvn(so.x_vec, mean_vec_bridge, so.cov_mat*sqrt_Δt, sde_mod.dim)
    prob_em[i_particle] += calc_log_pdf_mvn(so.x_vec, mean_vec_em, so.beta_mat*sqrt_Δt, sde_mod.dim)
    
end


"""
    propegate_modified_diffusion!(x, c, t_step_info, sde_mod, n_particles, u)

Propegate n-particles in the modified diffusion bridge filter for a SDE-model. 

Propegates n-particles for an individual with parameter vector c between 
time-points t_step_info[1] and t_step_info[2] using t_step_info[3] steps 
betweeen the time-points. Old particle values x are overwritten for memory 
efficiency. Negative values are set to 0 to avoid negative square-roots. 
The auxillerary variables contain random normal numbers used to propegate, 
and the solver_obj contains pre-allocated matrices and vectors. 
"""
function propegate_modified_diffusion!(x::Array{Float64, 2}, 
                                       mod_param::ModelParameters,
                                       t_step_info::TimeStepInfo, 
                                       sde_mod::SdeModel, 
                                       n_particles::T1, 
                                       u::Array{Float64, 2}, 
                                       y_vec_obs, 
                                       prob_bridge_log::Array{Float64, 1}, 
                                       prob_em_log::Array{Float64, 1}, 
                                       solver_obj::DiffBridgeSolverObj) where {T1<:Signed}
    
    p::DynModInput = mod_param.individual_parameters
    error_parameters::Array{Float64, 1} = mod_param.error_parameters
    # Stepping options for the EM-stepper
    Δt::Float64 = (t_step_info.t_end - t_step_info.t_start) / t_step_info.n_step
    t_end::Float64 = t_step_info.t_end
    t_vec = t_step_info.t_start:Δt:t_step_info.t_end
    solver_obj.Δt[1] = Δt
    solver_obj.sqrt_Δt[1] = sqrt(Δt)
    solver_obj.sigma_mat[diagind(solver_obj.sigma_mat)] .= error_parameters.^2

    # Update each particle (note x is overwritten)
    n_states = 1:sde_mod.dim
    @inbounds for i in 1:n_particles
        i_acc = 1:sde_mod.dim
        prob_em_log[i] = 0.0
        prob_bridge_log[i] = 0.0
        solver_obj.x_vec .= x[:, i]
        @inbounds for j in 1:t_step_info.n_step     
            u_vec = @view u[i_acc, i] 

            t::Float64 = t_vec[j]
            delta_k::Float64 = t_end - t

            # Update vectors for propegation 
            modified_diffusion_calc_arrays!(p, sde_mod, solver_obj, delta_k, y_vec_obs, t) 

            # Propegate and update probability vectors 
            modified_diffusion_propegate!(prob_em_log, prob_bridge_log, solver_obj, u_vec, i, sde_mod)
            i_acc = i_acc .+ sde_mod.dim
            
        end
        x[:, i] .= solver_obj.x_vec
    end
end


"""
    run_filter(filt_opt::ModifedDiffusionBridgeFilter,
               model_parameters::ModelParameters, 
               random_numbers::RandomNumbers, 
               sde_mod::SdeModel, 
               individual_data::IndData)::Float64

Run bootstrap filter for modified diffusion bridge SDE-stepper to obtain unbiased likelihood estimate. 

Each filter takes the input filt_opt, model-parameter, random-numbers, model-struct and 
individual_data. The filter is optmised to be fast and memory efficient on a single-core. 

# Args
- `filt_opt`: filter options (ModifedDiffusionBridgeFilter-struct)
- `model_parameters`: none-transfmored unknown model-parameters (ModelParameters)
- `random_numbers`: auxillerary variables, random-numbers, used to estimate the likelihood (RandomNumbers-struct)
- `sde_mod`: underlaying SDE-model for calculating likelihood (SdeModel struct)
- `individual_data`: observed data, and number of time-steps to perform between data-points (IndData-struct)

See also: [`ModifedDiffusionBridgeFilter`, `ModelParameters`, `RandomNumbers`, `SdeModel`, `IndData`]
"""
function run_filter(filt_opt::ModifedDiffusionBridgeFilter,
                    model_parameters::ModelParameters, 
                    random_numbers::RandomNumbers, 
                    sde_mod::SdeModel, 
                    individual_data::IndData)::Float64

    # Nested function that updates the weights (normalised and non-normalised)
    # for the modified diffusion bridge filter. Note, weights are calculated on 
    # log-scale for keeping precision. 
    @inline function calc_weights!(i_t_vec)::Float64

        y_obs_sub = SubArray(y_mat, (i_dim_obs, i_t_vec))

        @inbounds for ix in 1:n_particles
            x_curr_sub = @view x_curr[:, ix] 
            sde_mod.calc_obs(y_mod_vec, x_curr_sub, c, t_vec[i_t_vec])
            prob_obs_log::Float64 = log(sde_mod.calc_prob_obs(y_obs_sub, y_mod_vec, error_param, t_vec[i_t_vec], sde_mod.dim_obs))

            log_w_unormalised = (prob_em_log[ix] + prob_obs_log) - prob_bridge_log[ix]
            w_unormalised[ix] = exp(log_w_unormalised)
        end

        sum_w_unormalised_ret::Float64 = sum(w_unormalised)
        w_normalised .= w_unormalised ./ sum_w_unormalised_ret
        return sum_w_unormalised_ret
    end

    # Extract individual parameters for propegation 
    n_particles::Int64 = filt_opt.n_particles
    c::DynModInput = model_parameters.individual_parameters
    error_param::Array{Float64, 1} = model_parameters.error_parameters

    # Extract individual data and discretization level between time-points 
    t_vec::Array{Float64, 1} = individual_data.t_vec
    y_mat::Array{Float64, 2} = individual_data.y_mat
    n_step_vec::Array{Int16, 1} = individual_data.n_step
    len_t_vec::Int64 = length(t_vec)
        
    # Pre-allocated variables required for looping in the filter
    x0_mat::Array{Float64, 2} = reshape(repeat(model_parameters.x0, n_particles), (sde_mod.dim, n_particles))
    # Calculate initial values for particles (states)
    for i in 1:n_particles
        sde_mod.calc_x0!((@view x0_mat[:, i]), model_parameters)
    end
    x_curr::Array{Float64, 2} = Array{Float64, 2}(undef, (sde_mod.dim, n_particles))
    x_curr .= x0_mat

    w_unormalised::Array{Float64, 1} = Array{Float64, 1}(undef, n_particles)
    w_normalised::Array{Float64, 1} = Array{Float64, 1}(undef, n_particles)
    prob_bridge_log::Array{Float64, 1} = zeros(Float64, n_particles) 
    prob_em_log::Array{Float64, 1} = zeros(Float64, n_particles) 
    y_mod_vec::Array{Float64, 1} = Array{Float64, 1}(undef, sde_mod.dim_obs)
    i_dim_obs = 1:sde_mod.dim_obs
    i_dim_mod = 1:sde_mod.dim
    n_particles_inv::Float64 = convert(Float64, 1 / n_particles)

    log_lik::Float64 = 0.0

    # If correlated-filter, convert standard-normal resampling numbers to 
    # standard uniform 
    if filt_opt.ρ != 0
        u_resamp_vec_tmp = deepcopy(random_numbers.u_resamp)
        u_resamp_vec_tmp = cdf(Normal(), u_resamp_vec_tmp)
    else
        u_resamp_vec_tmp = deepcopy(random_numbers.u_resamp)
    end
    u_resamp_vec::Array{Float64, 1} = u_resamp_vec_tmp

    # Propegate particles for t1 
    i_u_prop::Int64 = 1  # Which discretization level to access 
    i_col_u_mat = 1:n_particles  # Which random numbers to use for propegation 
        
    # Solver object for diffusion bridge (avoid allocaiton 
    solver_obj::DiffBridgeSolverObj = init_sol_object(Val(sde_mod.dim), Val(sde_mod.dim_obs), sde_mod)
    
    # Special case where t = 0 is not observed 
    y_vec_obs = zeros(MVector{sde_mod.dim_obs, Float64})
    if t_vec[1] > 0.0
        # Extract random numbers for propegation. 
        t_step_info = TimeStepInfo(0.0, t_vec[1], n_step_vec[i_u_prop])
        y_vec_obs .= y_mat[:, 1]
        
        try
            propegate_modified_diffusion!(x_curr, 
                                    model_parameters,
                                    t_step_info, 
                                    sde_mod, 
                                    n_particles, 
                                    random_numbers.u_prop[i_u_prop], 
                                    y_vec_obs, 
                                    prob_bridge_log, 
                                    prob_em_log, 
                                    solver_obj)                                       
        catch 
            return -Inf 
        end

        i_u_prop += 1
    end


    # Note, particles can be updated normally. If propagation did not occur 
    # then prob_bridge_log and prob_em_log are zero arrays 
    sum_w_unormalised::Float64 = calc_weights!(1)
    log_lik += log(sum_w_unormalised * n_particles_inv)

    # Indices for resampling 
    i_resamp::Array{UInt32, 1} = Array{UInt32, 1}(undef, n_particles)

    # Propegate over remaning time-steps 
    for i_t_vec in 2:1:len_t_vec    

        # If correlated, sort x_curr
        if filt_opt.ρ != 0
            data_sort = sum(x_curr.^2, dims=1)[1, :]
            i_sort = sortperm(data_sort)
            x_curr = x_curr[:, i_sort]
            w_normalised = w_normalised[i_sort]
        end

        u_resample = u_resamp_vec[i_t_vec-1]
        systematic_resampling!(i_resamp, w_normalised, n_particles, u_resample)
        x_curr = x_curr[:, i_resamp]

        # Variables for propeating correct particles  
        t_step_info = TimeStepInfo(t_vec[i_t_vec-1], t_vec[i_t_vec], n_step_vec[i_u_prop])
        y_vec_obs .= y_mat[:, i_t_vec]
                                     
        try 
            propegate_modified_diffusion!(x_curr, 
                                    model_parameters,
                                    t_step_info, 
                                    sde_mod, 
                                    n_particles, 
                                    random_numbers.u_prop[i_u_prop], 
                                    y_vec_obs, 
                                    prob_bridge_log, 
                                    prob_em_log, 
                                    solver_obj)  
        catch 
            return -Inf 
        end
        i_u_prop += 1

        # Update weights and calculate likelihood
        sum_w_unormalised = calc_weights!(i_t_vec)
        log_lik += log(sum_w_unormalised * n_particles_inv)
    end

    return log_lik

end

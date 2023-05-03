"""
    calc_norm_squared(x)

Calculate squared L2-norm of a vector x
"""
function calc_norm_squared(x)
    return sum(x.^2)
end


"""
    step_em_bootstrap!(p::DynModInput, 
                       so::BootstrapSolverObj, 
                       sde_mod::SdeModel,
                       u_vec, 
                       t::Float64) where T1<:Signed 

Propegate the particles one time-step for the Euler-Maruyama bootstrap filter. 

See also: [`propegate_em_bootstrap!`]
"""
function step_em_bootstrap!(filter_cache::BootstrapFilterEMCache, 
                            p::DynModInput, 
                            sde_mod::S,
                            Δt::Float64,
                            sqrt_Δt::Float64,
                            t::Float64) where {S<:SdeModel}

    # Calculate beta and alpha arrays 
    sde_mod.calc_alpha(filter_cache.alpha, filter_cache.x, p, t)
    sde_mod.calc_beta(filter_cache.beta, filter_cache.x, p, t)
    calc_cholesky!(filter_cache.beta, sde_mod.dim)
    filter_cache.x .+= filter_cache.alpha .* Δt .+ filter_cache.beta * filter_cache.u .* sqrt_Δt
end


"""
    propegate_em_bootstrap!(x::Array{Float64, 2}, 
                            p::DynModInput, 
                            solver_obj::BootstrapSolverObj,
                            t_step_info::TimeStepInfo, 
                            sde_mod::SdeModel, 
                            n_particles::T1, 
                            u::Array{Float64, 2}) where {T1<:Signed}

Propegate n-particles (x) in the bootstrap filter for a SDE-model using Euler-Maruyama stepper. 

Propegates n-particles for an individual with parameters p between time-points t_step_info.t_start 
and t_step_info.t_end using t_step_info.n_step. Old particle values x are overwritten for memory 
efficiency. Negative values are set to 0 to avoid negative square-roots. The auxillerary variables 
contain random normal numbers used to propegate, and the solver_obj contains pre-allocated 
matrices and vectors. 
"""
function propegate_em_bootstrap!(filter_cache::BootstrapFilterEMCache, 
                                 p::DynModInput, 
                                 t_step_info::TimeStepInfo, 
                                 sde_mod::SdeModel, 
                                 n_particles::Int64, 
                                 u::Matrix{Float64}) 
    
    # Stepping options for the EM-stepper
    Δt::Float64 = (t_step_info.t_end - t_step_info.t_start) / t_step_info.n_step
    sqrt_Δt = sqrt(Δt)
    t_vec = t_step_info.t_start:Δt:t_step_info.t_end
    
    # Update each particle (note x is overwritten)
    @inbounds for i in 1:n_particles
        i_acc = 1:sde_mod.dim
        @views filter_cache.x .= filter_cache.particles[:, i]
        
        @inbounds for j in 1:t_step_info.n_step
            
            @views filter_cache.u .= u[i_acc, i] 
            step_em_bootstrap!(filter_cache, p, sde_mod, Δt, sqrt_Δt, t_vec[j])

            map_to_zero!(filter_cache.x, sde_mod.dim)

            i_acc = i_acc .+ sde_mod.dim
        end

        filter_cache.particles[:, i] .= filter_cache.x
    end

end


"""
    run_filter(filt_opt::BootstrapFilterEM,
               model_parameters::ModelParameters, 
               random_numbers::RandomNumbers, 
               sde_mod::SdeModel, 
               individual_data::IndData)::Float64

Run bootstrap filter for Euler-Maruyama SDE-stepper to obtain unbiased likelihood estimate. 

Each filter takes the input filt_opt, model-parameter, random-numbers, model-struct and 
individual_data. The filter is optmised to be fast and memory efficient on a single-core. 

# Args
- `filt_opt`: filter options (BootstrapFilterEM-struct)
- `model_parameters`: none-transfmored unknown model-parameters (ModelParameters)
- `random_numbers`: auxillerary variables, random-numbers, used to estimate the likelihood (RandomNumbers-struct)
- `sde_mod`: underlaying SDE-model for calculating likelihood (SdeModel struct)
- `individual_data`: observed data, and number of time-steps to perform between data-points (IndData-struct)

See also: [`BootstrapFilterEM`, `ModelParameters`, `RandomNumbers`, `SdeModel`, `IndData`]
"""
function run_filter(filt_opt::BootstrapFilterEM,
                    model_parameters::ModelParameters, 
                    random_numbers::RandomNumbers,
                    filter_cache::BootstrapFilterEMCache, 
                    sde_mod::SdeModel, 
                    individual_data::IndData, 
                    ::Val{is_correlated})::Float64 where {is_correlated}

    # Extract individual parameters for propegation 
    n_particles::Int64 = filt_opt.n_particles
    c::DynModInput = model_parameters.individual_parameters
    error_param::Vector{Float64} = model_parameters.error_parameters

    # Extract individual data and discretization level between time-points 
    t_vec::Vector{Float64} = individual_data.t_vec
    y_mat::Matrix{Float64} = individual_data.y_mat
    n_step_vec::Vector{Int16} = individual_data.n_step
    len_t_vec::Int64 = length(t_vec)
    n_particles_inv::Float64 = 1.0 / n_particles
    log_lik::Float64 = 0.0
    
    # Calculate initial values for particles (states)
    @inbounds for i in 1:n_particles
        sde_mod.calc_x0!((@view filter_cache.particles[:, i]), model_parameters)
    end
    
    u_resample::Vector{Float64} = is_correlated ? cdf(Normal(), random_numbers.u_resamp) : random_numbers.u_resamp

    # Propegate particles for t1 
    i_u_prop::Int64 = 1  # Which discretization level to access 
    
    # Special case where t = 0 is not observed 
    if t_vec[1] > 0.0
        t_step_info = TimeStepInfo(0.0, t_vec[1], n_step_vec[i_u_prop])
        propegate_em_bootstrap!(filter_cache, c, t_step_info, sde_mod, n_particles, random_numbers.u_prop[i_u_prop])
        try 
            #propegate_em_bootstrap!(filter_cache, c, t_step_info, sde_mod, n_particles, random_numbers.u_prop[i_u_prop])
        catch 
            return -Inf 
        end
        i_u_prop += 1
    end

    # Update likelihood first time
    sum_w_unormalised::Float64 = calc_weights_bootstrap!(1, filter_cache, c, t_vec[1], error_param, sde_mod, y_mat, n_particles)
    log_lik += log(sum_w_unormalised * n_particles_inv)

    # Propegate over remaning time-steps 
    for i_t_vec in 2:1:len_t_vec    
        
        # If correlated, sort x_curr
        if is_correlated
            data_sort = sum(filter_cache.particles.^2, dims=1)[1, :]
            i_sort = sortperm(data_sort)
            filter_cache.particles .= filter_cache.particles[:, i_sort]
            filter_cache.w_normalised .= filter_cache.w_normalised[i_sort]
        end

        _u_resample = u_resample[i_t_vec-1]
        systematic_resampling!(filter_cache.index_resample, filter_cache.w_normalised, n_particles, _u_resample)
        filter_cache.particles .= filter_cache.particles[:, filter_cache.index_resample]
        
        # Variables for propeating correct particles  
        t_step_info = TimeStepInfo(t_vec[i_t_vec-1], t_vec[i_t_vec], n_step_vec[i_u_prop])   
        try 
            propegate_em_bootstrap!(filter_cache, c, t_step_info, sde_mod, n_particles, random_numbers.u_prop[i_u_prop])         
        catch 
            return -Inf 
        end
        i_u_prop += 1
        
        # Update weights and calculate likelihood
        sum_w_unormalised = calc_weights_bootstrap!(i_t_vec, filter_cache, c, t_vec[i_t_vec], error_param, sde_mod, y_mat, n_particles)
        log_lik += log(sum_w_unormalised * n_particles_inv)
    end

    return log_lik
end


function calc_weights_bootstrap!(i_t_vec::Int64, 
                                 filter_cache::BootstrapFilterEMCache,                                 
                                 c,
                                 t::Float64,
                                 error_param::Vector{Float64},
                                 sde_mod::SdeModel,
                                 y_mat::Matrix{Float64}, 
                                 n_particles::Int64)::Float64 

    y_obs_sub = @view y_mat[1:sde_mod.dim_obs, i_t_vec]
    @inbounds for i in 1:n_particles
        x = @view filter_cache.particles[:, i] 
        sde_mod.calc_obs(filter_cache.y_model, x, c, t)
        filter_cache.w_unormalised[i] = sde_mod.calc_prob_obs(y_obs_sub, filter_cache.y_model, error_param, t, sde_mod.dim_obs)
    end
    sum_w_unormalised = sum(filter_cache.w_unormalised)
    filter_cache.w_normalised .= filter_cache.w_unormalised ./ sum_w_unormalised

    return sum_w_unormalised
end
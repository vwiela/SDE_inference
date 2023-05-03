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
    modified_diffusion_propegate!(prob_em::Array{Float64, 1}, 
                                  prob_bridge::Array{Float64, 1}, 
                                  so::DiffBridgeSolverObj, 
                                  u_vec, 
                                  i_particle::T1, 
                                  sde_mod::SdeModel) where T1<:Signed

Propegate the particles one time-step for the modified diffusion bridge. 
"""
function modified_diffusion_propegate!(filter_cache::ModifiedBridgeFilterCache, 
                                       p,
                                       t::Float64,
                                       Δt::Float64,
                                       sqrt_Δt::Float64,
                                       Δk::Float64,
                                       i_particle::Int64, 
                                       sde_mod::SdeModel) 

    # Calculate beta and alpha arrays 
    sde_mod.calc_alpha(filter_cache.alpha, filter_cache.x, p, t)
    sde_mod.calc_beta(filter_cache.beta, filter_cache.x, p, t)

    # Compute mean and covariance matrix for bridge 
    inv_term = filter_cache.beta*filter_cache.P*inv(filter_cache.P_T*filter_cache.beta*filter_cache.P*Δk + filter_cache.Σ)
    filter_cache.μ .= filter_cache.alpha .+ inv_term * (filter_cache.y_obs - filter_cache.P_T * (filter_cache.x + filter_cache.alpha * Δk))
    filter_cache.Ω .= filter_cache.beta .- inv_term*filter_cache.P_T*filter_cache.beta*Δt                        

    # Must calculate mean-vectors before propegating (when calculcating logpdf)
    @. filter_cache.μ_bridge_pdf = filter_cache.x + filter_cache.μ*Δt
    @. filter_cache.μ_EM_pdf = filter_cache.x + filter_cache.alpha*Δt

    # Note, for propegation and then log-pdf cholesky decompositon is required for cov-mat and beta 
    calc_cholesky!(filter_cache.Ω, sde_mod.dim)
    calc_cholesky!(filter_cache.beta, sde_mod.dim)

    # Propegate
    filter_cache.x .+= filter_cache.μ*Δt + filter_cache.Ω*filter_cache.u * sqrt_Δt
    map_to_zero!(filter_cache.x, sde_mod.dim)
    
    # Update probabilities, cov_mat and beta-mat are both lower-triangular cholesky
    filter_cache.Ω .*= sqrt_Δt
    filter_cache.beta .*= sqrt_Δt
    filter_cache.logpdf_bridge[i_particle] += calc_log_pdf_mvn(filter_cache.x, filter_cache.μ_bridge_pdf, filter_cache.Ω, sde_mod.dim)
    filter_cache.logpdf_EM[i_particle] += calc_log_pdf_mvn(filter_cache.x, filter_cache.μ_EM_pdf, filter_cache.beta, sde_mod.dim)
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
function propegate_modified_diffusion!(filter_cache::ModifiedBridgeFilterCache,
                                       mod_param::ModelParameters,
                                       t_step_info::TimeStepInfo, 
                                       sde_mod::SdeModel, 
                                       n_particles::Int64, 
                                       u::Matrix{Float64}) 
    
    p::DynModInput = mod_param.individual_parameters
    filter_cache.Σ[diagind(filter_cache.Σ)] .= mod_param.error_parameters.^2
    
    # Time-stepping options 
    Δt::Float64 = (t_step_info.t_end - t_step_info.t_start) / t_step_info.n_step
    sqrt_Δt::Float64 = sqrt(Δt)
    t_end::Float64 = t_step_info.t_end
    t_vec = t_step_info.t_start:Δt:t_step_info.t_end

    fill!(filter_cache.logpdf_bridge, 0.0)
    fill!(filter_cache.logpdf_EM, 0.0)
    
    # Update each particle (note x is overwritten)
    @inbounds for i in 1:n_particles
        i_acc = 1:sde_mod.dim
        @views filter_cache.x .= filter_cache.particles[:, i]
        for j in 1:t_step_info.n_step     
            @views filter_cache.u .= u[i_acc, i] 
            t = t_vec[j]
            Δk = t_end - t
            # Propegate and update probability vectors 
            modified_diffusion_propegate!(filter_cache, p, t, Δt, sqrt_Δt, Δk, i, sde_mod)
            i_acc = i_acc .+ sde_mod.dim
        end
        filter_cache.particles[:, i] .= filter_cache.x
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
                    filter_cache::ModifiedBridgeFilterCache, 
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
        
    # Calculate initial values for particles (states)
    @inbounds for i in 1:n_particles
        sde_mod.calc_x0!((@view filter_cache.particles[:, i]), model_parameters)
    end
    

    n_particles_inv::Float64 = 1.0 / n_particles
    log_lik::Float64 = 0.0
    u_resample::Vector{Float64} = is_correlated ? cdf(Normal(), random_numbers.u_resamp) : random_numbers.u_resamp
    i_u_prop::Int64 = 1  
    
    if t_vec[1] > 0.0
        # Extract random numbers for propegation. 
        t_step_info = TimeStepInfo(0.0, t_vec[1], n_step_vec[i_u_prop])
        @views filter_cache.y_obs .= y_mat[:, 1]
        try
            propegate_modified_diffusion!(filter_cache,
                                          model_parameters,
                                          t_step_info, 
                                          sde_mod, 
                                          n_particles, 
                                          random_numbers.u_prop[i_u_prop])                                       
        catch 
            return -Inf 
        end

        i_u_prop += 1
    end


    # Note, particles can be updated normally. If propagation did not occur 
    # then prob_bridge_log and prob_em_log are zero arrays 
    sum_w_unormalised::Float64 = calc_weights_modifed_bridge!(1, filter_cache, c, t_vec[1], error_param, sde_mod, y_mat, n_particles)
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
        @views filter_cache.y_obs .= y_mat[:, i_t_vec]
        try 
            propegate_modified_diffusion!(filter_cache,
                                          model_parameters,
                                          t_step_info, 
                                          sde_mod, 
                                          n_particles, 
                                          random_numbers.u_prop[i_u_prop])                                       
        catch 
            return -Inf 
        end
        i_u_prop += 1

        # Update weights and calculate likelihood
        sum_w_unormalised = calc_weights_modifed_bridge!(i_t_vec, filter_cache, c, t_vec[i_t_vec], error_param, sde_mod, y_mat, n_particles)
        log_lik += log(sum_w_unormalised * n_particles_inv)
    end

    return log_lik
end


function calc_weights_modifed_bridge!(i_t_vec::Int64, 
                                      filter_cache::ModifiedBridgeFilterCache,                                 
                                      c,
                                      t::Float64,
                                      error_param::Vector{Float64},
                                      sde_mod::SdeModel,
                                      y_mat::Matrix{Float64}, 
                                      n_particles::Int64)::Float64 

    y_obs = @view y_mat[1:sde_mod.dim_obs, i_t_vec]
    @inbounds for ix in 1:n_particles
        x = @view filter_cache.particles[:, ix] 
        sde_mod.calc_obs(filter_cache.y_model, x, c, t)
        prob_obs_log = log(sde_mod.calc_prob_obs(y_obs, filter_cache.y_model, error_param, t, sde_mod.dim_obs))
        filter_cache.w_unormalised[ix] = exp((filter_cache.logpdf_EM[ix] + prob_obs_log) - filter_cache.logpdf_bridge[ix])
    end

    sum_w_unormalised::Float64 = sum(filter_cache.w_unormalised)
    filter_cache.w_normalised .= filter_cache.w_unormalised ./ sum_w_unormalised
    return sum_w_unormalised
end
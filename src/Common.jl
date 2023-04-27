#= 
    Common functions used by all filters. 

    Create ind-data struct 
    Create and update random numbers 
    Create model-parameters struct 
    Init filter struct 
    Change option for filter struct 
=# 


"""
    create_n_step_vec(t_vec, delta_t)

Calculate number of time-steps for EM-solver between each observed time-point 

The discretization level (delta_t) is provided by the user, and should be 
small enough to ensure accuracy, while keeping computational effiency. 
"""
function create_n_step_vec(t_vec, delta_t)
    # Adapt length of n_step if t[1] != 0
    len_step = 0
    zero_in_t = false
    if t_vec[1] == 0
        len_step = length(t_vec) - 1
        zero_in_t = true
    else
        len_step = length(t_vec)
    end
    n_step::Array{Int16, 1} = Array{Int16, 1}(undef, len_step)
    i_n_step = 1

    # Special case where t = 0 is not observed
    if !zero_in_t
        n_step[1] = convert(Int16, round((t_vec[1] - 0.0) / delta_t))
        i_n_step += 1
    end

    # Fill the step-vector
    for i_t_vec in 2:length(t_vec)
        n_steps = round((t_vec[i_t_vec] - t_vec[i_t_vec-1]) / delta_t)
        n_step[i_n_step] = convert(Int16, n_steps)
        i_n_step += 1
    end

    # Ensure correct type 
    n_step = convert(Array{Int16, 1}, n_step)

    return n_step
end


"""
    init_ind_data(path::String, delta_t::Float64)

Create IndData-struct using provided path to data-file and time-discretization. 

The data-file in path must be a csv-file in tidy-format. 

See also: [`IndData`](@ref)
"""
function init_ind_data(data_obs::DataFrame, filter_opt; cov_name::T1=[""]) where T1<:Array{<:String, 1}

    
    delta_t = filter_opt.delta_t
    t_vec::Array{Float64, 1} = deepcopy(convert(Array{Float64, 1}, data_obs[!, "time"]))
    unique!(t_vec); sort!(t_vec)

    # General parmeters for the data set 
    n_data_points = length(t_vec)
    len_dim_obs_model = length(unique(data_obs[!, "obs_id"]))
    y_mat::Array{Float64, 2} = fill(convert(Float64, NaN), (len_dim_obs_model, n_data_points))

    i_obs = unique(data_obs[!, "obs_id"])

    # Fill y_mat for each obs id 
    for i in 1:len_dim_obs_model
        i_data_obs_i = (i_obs[i] .== data_obs[!, "obs_id"])
        t_vec_obs = data_obs[!, "time"]
        y_vec_obs = data_obs[!, "obs"]

        y_vec_obs = y_vec_obs[i_data_obs_i]
        t_vec_obs = t_vec_obs[i_data_obs_i]

        # Populate observation matrix 
        for j in eachindex(t_vec_obs)
            i_t = findall(x->x==t_vec_obs[j], t_vec)[1]
            y_mat[i, i_t] = y_vec_obs[j]
        end

    end

    if cov_name[1] != ""
        cov_vals = Array{Float64, 1}(undef, length(cov_name))
        for i in eachindex(cov_name)
            cov_vals[i] = data_obs[1, cov_name[i]]        
        end
    else
        cov_vals = Array{Float64, 1}(undef, 0)
    end 

    ind_data = IndData(t_vec, y_mat, create_n_step_vec(t_vec, delta_t), cov_vals)

    return ind_data
end


"""
    create_rand_num(ind_data::IndData, sde_mod::SdeModel, filter; rng::MersenneTwister=MersenneTwister())

Allocate and initalise RandomNumbers-struct for a SDE-model particle filter. 

Ind_data object provides the number of time-step between each observed 
time-point. If non-correlated particles are used, resampling numbers 
are standard uniform, else standard normal. 

See also: [`RandomNumbers`](@ref)
"""
function create_rand_num(ind_data::IndData, sde_mod::SdeModel, filter; rng::MersenneTwister=MersenneTwister())
    # Allocate propegating particles
    n_particles = filter.n_particles
    dist_prop = Normal()
    len_u = length(ind_data.n_step)
    u_prop::Array{Array{Float64, 2}, 1} = Array{Array{Float64, 2}, 1}(undef, len_u)
    for i in 1:len_u
        n_row = convert(Int64, sde_mod.dim*ind_data.n_step[i])
        n_col = n_particles
        u_mat = randn(rng, Float64, (n_row, n_col))
        
        u_prop[i] = u_mat
    end

    # Numbers used for systematic resampling
    if filter.rho == 0
        u_sys_res = rand(rng, length(ind_data.t_vec))
    elseif filter.rho != 0
        u_sys_res = randn(rng, Float64, length(ind_data.t_vec))
    end
    
    return RandomNumbers(u_prop, u_sys_res)
end


"""
    update_random_numbers!(rand_num::RandomNumbers, 
                           filter::BootstrapFilterEm;
                           rng::MersenneTwister=MersenneTwister())

    
Update random numbers, auxillerary variables, (propegation and resampling) for the Boostrap EM-filter. 

If the particles are correlated particles u are updated via a Crank-Nichelson scheme. 

Overwrites the rand_num with the new random-numbers. 
"""
function update_random_numbers!(rand_num::RandomNumbers, 
                                filter::BootstrapFilterEm;
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::Float64 = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            dim = size(rand_num.u_prop[i])
            rand_num.u_prop[i] .= (filter.rho .* rand_num.u_prop[i]) + randn(rng, Float64, dim) *std 
        end

        dim = size(rand_num.u_resamp)
        rand_num.u_resamp .= (filter.rho .* rand_num.u_resamp) + randn(rng, Float64, dim) * std
    end
end
"""
    update_random_numbers!(rand_num_new::RandomNumbers, 
                           rand_num_old::RandomNumbers, 
                           filter::BootstrapFilterEm;
                           rng::MersenneTwister=MersenneTwister())

Using rand_num_old the new-random numbers are stored in rand_num_new.
"""
function update_random_numbers!(rand_num_new::RandomNumbers, 
                                rand_num_old::RandomNumbers, 
                                filter::BootstrapFilterEm;
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num_new.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num_new.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num_new.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::Float64 = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            randn!(rand_num_new.u_prop[i])
            rand_num_new.u_prop[i] .*= std
            rand_num_new.u_prop[i] .+= filter.rho .* rand_num_old.u_prop[i]
        end

        randn!(rand_num_new.u_resamp)
        rand_num_new.u_resamp .*= std
        rand_num_new.u_resamp .+= filter.rho .* rand_num_old.u_resamp
    end
end
"""
    update_random_numbers!(rand_num::RandomNumbers, 
                           filter::ModDiffusionFilter;
                           rng::MersenneTwister=MersenneTwister())

    
Update random numbers, auxillerary variables, (propegation and resampling) for the modified diffusion bridge 

Same approach as bootstrap EM. 
"""
function update_random_numbers!(rand_num::RandomNumbers, 
                                filter::ModDiffusionFilter, 
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::Float64 = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            dim = size(rand_num.u_prop[i])
            rand_num.u_prop[i] .= (filter.rho .* rand_num.u_prop[i]) + randn(rng, Float64, dim) *std 
        end

        dim = size(rand_num.u_resamp)
        rand_num.u_resamp .= (filter.rho .* rand_num.u_resamp) + randn(rng, Float64, dim) * std
    end
end
"""
    update_random_numbers!(rand_num_new::RandomNumbers, 
                           rand_num_old::RandomNumbers, 
                           filter::ModDiffusionFilter;
                           rng::MersenneTwister=MersenneTwister())
"""
function update_random_numbers!(rand_num_new::RandomNumbers, 
                                rand_num_old::RandomNumbers, 
                                filter::ModDiffusionFilter;
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num_new.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num_new.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num_new.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::Float64 = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            randn!(rng, rand_num_new.u_prop[i])
            rand_num_new.u_prop[i] .*= std
            rand_num_new.u_prop[i] .+= filter.rho .* rand_num_old.u_prop[i]
        end

        randn!(rng, rand_num_new.u_resamp)
        rand_num_new.u_resamp .*= std
        rand_num_new.u_resamp .+= filter.rho .* rand_num_old.u_resamp
    end
end


"""
    init_model_parameters(ind_param, error_param, model::SdeModel; covariates=false, kappa=false)::ModelParameters

Allocate and initalise ModelParameters-struct for sde-model (initial values Float64)

Individual parameters correspond to rate-constants and/or initial values. 
Error_param is assumed to be an array.

See also: [`ModelParameters`](@ref)
""" 
function init_model_parameters(ind_param, error_param, model::SdeModel; covariates=false, kappa=false)::ModelParameters
    
    if covariates == false
        covariates = Array{Float64, 1}(undef, 0)
    end
    if kappa == false
        kappa = Array{Float64, 1}(undef, 0)
        ind_param = DynModInput(ind_param, kappa, covariates)
    else
        ind_param = DynModInput(ind_param, kappa, covariates)
    end

    x0 = Array{Float64, 1}(undef, model.dim)
    #model.calc_x0!(x0, ind_param)

    # Struct object 
    return ModelParameters(ind_param, x0, error_param, covariates)
end


"""
    systematic_resampling!(index_sampling, weights, n_samples, u)

Calculate indices from systematic resampling of non-normalised weights. 

u must be standard uniform. 
"""
function systematic_resampling!(index_sampling::Array{UInt32, 1}, weights::Array{Float64, 1}, n_samples::T1, u::Float64) where T1<:Signed

    # For resampling u ~ U(0, 1/N)
    u /= n_samples
    # Step-length when traversing the cdf
    delta_u::Float64 = 1.0 / n_samples
    # For speed (avoid division)
    sum_weights_inv::Float64 = 1.0 / sum(weights)

    # Sample from the cdf starting from the random number
    k::UInt32 = 1
    sum_cum::Float64 = weights[k] * sum_weights_inv
    @inbounds for i in 1:n_samples
        # See which part of the distribution u is at
        while sum_cum < u
            k += 1
            sum_cum += weights[k] * sum_weights_inv
        end
        index_sampling[i] = k
        u += delta_u
    end
end


"""
    init_filter(filter::BootstrapEm, dt; n_particles=1000, rho=0.0) 

Initialise bootstrap particle filter struct for a SDE-model using Euler-Maruyama-stepper. 

By default a non-correlated particle filter is used (rho = 0.0). Step-length for 
Euler-Maruyama should be as large as possible, while still ensuring numerical accuracy. 

# Args
- `filter`: particle filter to use, BootstrapEm() = Bootstrap filter with Euler-Maruyama stepper
- `dt`: Euler-Maruyama step-length 
- `n_particles`: number of particles to use when estimating the likelihood
- `rho`: correlation level. rho ∈ [0.0, 1.0) and if rho = 0.0 the particles are uncorrelated. 
"""
function init_filter(filter::BootstrapEm,
                     dt;
                     n_particles=1000,
                     rho=0.0) 
    
    # Ensure correct type in calculations 
    dt = convert(Float64, dt)
    rho = convert(Float64, rho)
    filter_opt = BootstrapFilterEm(dt, n_particles, rho)
    return filter_opt
end
"""
    init_filter(filter::ModDiffusion, dt; n_particles=1000, rho=0.0) 

Initialise modified diffusion bridge filter for a SDE-model. 

By default a non-correlated particle filter is used (rho = 0.0). Step-length for the bridge
should be as large as possible, while still ensuring numerical accuracy. 

# Args
- `filter`: particle filter to use, ModDiffusion() = Modified diffusion bridge for SDE-models 
- `dt`: Modified diffusion bridge step-length 
- `n_particles`: number of particles to use when estimating the likelihood
- `rho`: correlation level. rho ∈ [0.0, 1.0) and if rho = 0.0 the particles are uncorrelated. 
"""
function init_filter(filter::ModDiffusion,
                     dt::Float64;
                     n_particles=1000,
                     rho=0.0) 
    
    # Ensure correct type in calculations 
    dt::Float64 = convert(Float64, dt)
    rho::Float64 = convert(Float64, rho)
    filter_opt = ModDiffusionFilter(dt, n_particles, rho)
    return filter_opt
end


"""
    change_filter_opt(filter::BootstrapFilterEm, n_particles, rho)

Create a deepcopy of a Bootstrap EM-filter with new number of particles and correlation-level rho.  

Used by pilot-run functions to investigate different particles. 
"""
function change_filter_opt(filter::BootstrapFilterEm, n_particles, rho)

    new_filter = BootstrapFilterEm(    
        filter.delta_t,
        n_particles,
        rho)
    new_filter = deepcopy(new_filter)

    return new_filter
end
"""
    change_filter_opt(filter::ModDiffusionFilter, n_particles::T1, rho::Float64) where T1<:Signed

Create a deepcopy of a modified-diffusion bridge filter with new number of particles and correlation-level rho.  
"""
function change_filter_opt(filter::ModDiffusionFilter, n_particles::T1, rho::Float64) where T1<:Signed

    new_filter = ModDiffusionFilter(    
        filter.delta_t,
        n_particles,
        rho)
    new_filter = deepcopy(new_filter)

    return new_filter
end



"""
    calc_log_jac(param::Array{Float64, 1}, param_pos::Array{Bool, 1}, n_param)

Calculate log-jacobian for parameters proposed via exponential-transformation. 

Param-array must be a one-dimensional array of equal length to param_pos which 
contains information of which parameters are enforced as positive. 
"""
function calc_log_jac(param, param_pos::Array{Bool, 1}, n_param::T)::Float64 where T<:Signed
    log_jac::Float64 = 0.0
    @simd for i in 1:n_param
        if param_pos[i] == true
            log_jac -= log(deepcopy(param[i]))
        end
    end

    return log_jac 
end


"""
    calc_log_prior(x, prior_dist::T1, n_param) where T1<:Array{<:Distribution, 1}

Calculate log-prior for array x based on the priors-distributions array of length n_param. 
"""
function calc_log_prior(x, prior_dist::T1, n_param::T2)::Float64 where {T1<:Array{<:Distribution, 1}, T2<:Signed}

    log_prior::Float64 = 0.0
    @simd for i in 1:n_param
        @inbounds log_prior += logpdf(prior_dist[i], x[i])
    end

    return log_prior
end


"""
    init_file_loc(path_data::String, model_name::String; 
        dir_save::String="", multiple_ind=false)

Initalise FileLocations-struct for a model. 

If the user does not provide a dir-save, the sampling results 
are stored in intermediate-directory (strongly recomended)

See also: [`FileLocations`](@ref)
"""
function init_file_loc(path_data::String, model_name::String; 
    dir_save::String="", multiple_ind=false, cov_name::T1=[""], 
    cov_val=Array{Float64, 1}(undef, 0), dist_id=ones(Int64, 0))::FileLocations where T1<:Array{<:String, 1}

    # Sanity ensure input 
    if model_name[1] == "/"
        model_name = model_name[2:end]
    end

    if dir_save == "" && multiple_ind == false
        dir_save = pwd() * "/Intermediate/Single_individual/" * model_name
    elseif dir_save == "" && multiple_ind == true 
        dir_save = pwd() * "/Intermediate/Multiple_individuals/" * model_name
    end

    if length(dist_id) != 0
        if length(dist_id) != length(cov_val)
            @printf("Error: Length of dist-id does not match the length of cov-val")
        end
    end
    if length(cov_val) == 0
        dist_id = ones(Int64, 1)
    end
    if length(cov_val) != 0 && length(dist_id) == 0
        dist_id = ones(Int64, length(cov_val))
    end


    file_loc = FileLocations(path_data, model_name, dir_save, cov_name, cov_val, dist_id)
    return file_loc
end


"""
    calc_dir_save!(file_loc::FileLocations, filter, mcmc_sampler)

For a specific filter and mcmc-sampler calculate correct sub-directory for saving result. 

Initalisation of file-locations creates the main-directory to save pilot-run data 
and inference result. Based on filter, mcmc-sampler this function creates the 
correct sub-directory for saving the result. User does not access this function. 

See also: [`FileLocations`, `init_file_loc`](@ref)
"""
function calc_dir_save!(file_loc::FileLocations, filter, mcmc_sampler; mult_ind=false)
    if filter.rho != 0
        tag_corr = "/Correlated"
    elseif filter.rho == 0
        tag_corr = "/Not_correlated"
    end

    if mcmc_sampler.name_sampler == "Gen_am_sampler"
        tag_samp = "/Gen_am_sampler"
    elseif mcmc_sampler.name_sampler == "Rand_walk_sampler"
        tag_samp = "/Rand_walk_sampler"
    elseif mcmc_sampler.name_sampler == "Am_sampler"
        tag_samp = "/Am_sampler"
    elseif mcmc_sampler.name_sampler == "Ram_sampler"
        tag_samp = "/Ram_sampler"
    end

    if mult_ind == true
        tag_mult = "/Multiple"
    else
        tag_mult = ""
    end

    # File-locations is mutable, this must be changed!
    file_loc.dir_save = file_loc.dir_save * tag_mult * tag_corr * tag_samp * "/"
end


"""
    calc_param_init_val(init_param, prior_list)::Array{Float64, 1}

Initial value for parameter-vector using mean, mode, median or random (init_param) on prior_list. 

Only works for case sensitive mean, mode, median or random for init_param. For cachy-distributions 
mean and mode are changed to median. 
"""
function calc_param_init_val(init_param::String, prior_list)::Array{Float64, 1}
    
    # If multivariate prior provided adapt length. Adapt for empty-prior-list
    len_prior = length(prior_list)
    if len_prior > 0
        if length(prior_list[1]) > 1
            len_prior = length(prior_list[1])
        end
    end

    if !(init_param == "mean" || init_param == "mode" || init_param == "random" || init_param == "median")
        @printf("Error: If init_param is a string it should be either ")
        @printf("mean, mode or random. Provided: %s\n", init_param)
        return 1 
    end

    # Change to median if Cauchy-distribution is provided
    for i in 1:length(prior_list)
        change_val = false
        if typeof(prior_list[i]) <: Truncated{<:Cauchy{<:AbstractFloat}, Continuous, <:AbstractFloat}
            change_val = true 
        elseif typeof(prior_list[i]) <: Cauchy{<:AbstractFloat}
            change_val = true 
        end

        if change_val == true && !(init_param == "median" || init_param == "random")
            @printf("As cauchy distribution is used will change to median for init-param\n")
            init_param = "median"
        end
    end

    # Calculate the number of parameters (account for multivariate distribution)
    if length(prior_list) == 1
        n_prior_dists = 1
        n_param = length(prior_list[1])
    else
        n_prior_dists = length(prior_list)
        n_param = n_prior_dists
    end

    param_init::Array{Float64, 1} = Array{Float64, 1}(undef, n_param)

    # In the case either mean, mode or random is provided 
    if init_param == "mean"
        param_init_tmp = mean.(prior_list)
    elseif init_param == "mode"
        param_init_tmp = mode.(prior_list)
    elseif init_param == "median"
        param_init_tmp = median.(prior_list)
    elseif init_param == "random"
        param_init_tmp = rand.(prior_list)
    end

    # Handle the case if multivariate distribution is sent in 
    if typeof(param_init_tmp) <: Array{<:Array{<:AbstractFloat, 1}, 1}
        param_init .= param_init_tmp[1]
    else
        param_init .= param_init_tmp
    end
    
    return param_init
end
"""
    calc_param_init_val(init_param::T, prior_list)::Array{Float64, 1} where T<:Array{<:AbstractFloat, 1}

Initial value for parameter-vector using provided array of equal length to prior-list. 
"""
function calc_param_init_val(init_param::T, prior_list)::Array{Float64, 1} where T<:Array{<:AbstractFloat, 1}

    # If multivariate prior provided adapt length. Adapt for empty-prior-list
    len_prior = length(prior_list)
    if len_prior > 0
        if length(prior_list[1]) > 1
            len_prior = length(prior_list[1])
        end
    end

    if length(init_param) != len_prior
        @printf("If providing start-guesses, the number of start-guesses ")
        @printf("must match the number of priors\n")
        return 1 
    end

    param_init::Array{Float64, 1} = Array{Float64, 1}(undef, len_prior)

    for i in 1:len_prior
        param_init[i] = convert(Float64, init_param[i])
    end

    return param_init
end


"""
    calc_bool_array(bool_val::Array{Bool, 1}, n_param::T1)::Array{Bool, 1} where T1 <: Signed

Check that bool-array has length n_param and return bool_val upon success. 
"""
function calc_bool_array(bool_val::Array{Bool, 1}, n_param::T1)::Array{Bool, 1} where T1 <: Signed
    
    # Sanity check input 
    if length(bool_val) != n_param
        @printf("Error: When creating bool-array from array it must ")
        @printf("match the number of parameters\n")
        return 1 
    end
   
    return bool_val

end
"""
    calc_bool_array(bool_val::Bool, n_param::T1)::Array{Bool, 1} where T1 <: Signed

Return bool-array of length n_param consting of bool-val. 
"""
function calc_bool_array(bool_val::Bool, n_param::T1)::Array{Bool, 1} where T1<:Signed

    bool_array::Array{Bool, 1} = Array{Bool, 1}(undef, n_param)
    bool_array .= bool_val

    return bool_array
end


"""
    empty_dist()::Array{Normal{Float64}, 1}

Intialise empty array of <:Array{Distribution, 1}. 
"""
function empty_dist()::Array{Normal{Float64}, 1}
    dist_use::Array{Normal{Float64}, 1} = deleteat!([Normal()], 1)
    return dist_use
end



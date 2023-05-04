"""
         init_param(prior_ind_param, 
                    prior_error_param; 
                    init_ind_param="mean", 
                    init_error_param="mean", 
                    ind_param_pos=true, 
                    error_param_pos=true, 
                    ind_param_log=true, 
                    error_param_log=true)

Initalise InitParameterInfo-struct for a single-individual inference model. 

Priors must be arrays of distributions defined by Distributions.jl. Init_ind_param and 
init_error_param can either be arrays (of starting values) or "mean", "mode" and 
"random". In the latter cases starting-values are generated from the priors. 
By defult, parameters are assumed to be postive, and not being estimated 
on the log-scale. In case of more fine-tuning, arrays of bools can be 
provided for the log-scale and positivity specification of the parameters. 

See also: [`InitParameterInfo`](@ref)
"""
function init_param(prior_ind_param, 
                    prior_error_param; 
                    init_ind_param="mean", 
                    init_error_param="mean", 
                    ind_param_pos=true, 
                    error_param_pos=true, 
                    ind_param_log=false, 
                    error_param_log=false)

    n_ind_param = length(prior_ind_param)
    n_error_param = length(prior_error_param)

    ind_0 = Array{Float64, 1}(undef, n_ind_param)
    ind_pos = Array{Bool, 1}(undef, n_ind_param)
    ind_log = Array{Bool, 1}(undef, n_ind_param)
    error_0 = Array{Float64, 1}(undef, n_error_param)
    error_pos = Array{Bool, 1}(undef, n_error_param)
    error_log = Array{Bool, 1}(undef, n_error_param)

    # Sanity check init_ind and init_error 
    if typeof(init_ind_param) == String
        if init_ind_param != "mean" && init_ind_param != "mean"
            print("Error: init_ind must, if not being an array,")
            print("be either mean or mode, provided:", init_ind, "\n")
            return 1
        end
    elseif typeof(init_ind_param) <: Array{<:AbstractFloat, 1} && length(init_ind_param) != n_ind_param
        print("Error: If init_ind is an array if must consist of floats")
        print("and be of same length as number of individual parameters.\n")
        return 1
    end

    if typeof(init_error_param) == String
        if init_error_param != "mean" && init_error_param != "mean"
            print("Error: init_error must, if not being an array,")
            print("be either mean or mode, provided:", init_error, "\n")
            return 1
        end
    elseif typeof(init_error_param) <: Array{<:AbstractFloat, 1} && length(init_error_param) != n_error_param
        print("Error: If error_ind_param is an array if must consist of floats")
        print("and be of same length as number of error parameters.\n")
        return 1
    end

    # Initalise error and individual parameters 
    if init_ind_param == "mean"
        for i in 1:n_ind_param
            ind_0[i] = mean(prior_ind_param[i])
        end
    elseif init_ind_param == "mode"
        for i in 1:n_ind_param
            ind_0[i] = mode(prior_ind_param[i])
        end 
    elseif typeof(init_ind_param) <: Array{<:AbstractFloat, 1}
        for i in 1:n_ind_param
            ind_0[i] = init_ind_param[i]
        end
    end

    if init_error_param == "mean"
        for i in 1:n_error_param
            error_0[i] = mean(prior_error_param[i])
        end
    elseif init_error_param == "mode"
        for i in 1:n_error_param
            error_0[i] = mode(prior_error_param[i])
        end 
    elseif typeof(init_error_param) <: Array{<:AbstractFloat, 1}
        for i in 1:n_error_param
            error_0[i] = init_error_param[i]
        end
    end

    # Sanity check the positivity input
    if (typeof(ind_param_pos) != Bool && 
        typeof(ind_param_pos) != Array{Bool, 1} && 
        length(ind_param_pos) != n_ind_param)

        print("Error: ind_param_pos should be an array of bools or")
        print("a simple bool\n")
        return 1 
    end

    if (typeof(error_param_pos) != Bool && 
        typeof(error_param_pos) != Array{Bool, 1} && 
        length(error_param_pos != n_error_param))

        print("Error: ind_param_pos should be an array of bools the length")
        print("of the error parameters or a simple bool\n")
        return 1 
    end

    # Fill the postive arrays 
    if typeof(ind_param_pos) == Bool
        ind_pos .= ind_param_pos
    else
        for i in 1:n_ind_param
            ind_pos[i] = ind_param_pos[i]
        end
    end

    if typeof(error_param_pos) == Bool
        error_pos .= error_param_pos
    else
        for i in 1:n_error_param
            error_pos[i] = error_param_pos[i]
        end
    end

    # Sanity check the log-input
    if (typeof(ind_param_log) != Bool && 
        typeof(ind_param_log) != Array{Bool, 1} && 
        length(ind_param_log) != n_ind_param)

        print("Error: ind_param_log should be an array of bools or")
        print("a simple bool\n")
        return 1 
    end

    if (typeof(error_param_log) != Bool && 
        typeof(error_param_log) != Array{Bool, 1} && 
        length(error_param_log) != n_error_param)

        print("Error: ind_param_pos should be an array of bools the length")
        print("of the error parameters or a simple bool\n")
        return 1 
    end

    # Fill the postive arrays 
    if typeof(ind_param_log) == Bool
        ind_log .= ind_param_log
    else
        for i in 1:n_ind_param
            ind_log[i] = ind_param_log[i]
        end
    end

    if typeof(error_param_log) == Bool
        error_log .= error_param_log
    else
        for i in 1:n_error_param
            error_log[i] = error_param_log[i]
        end
    end

    n_param = n_error_param + n_ind_param
    init_param_info = InitParameterInfo(prior_ind_param, prior_error_param, 
                                        ind_0, error_0, n_param,
                                        ind_pos, error_pos, 
                                        ind_log, error_log)

    return init_param_info 
end 


"""
    map_proposal_to_model_parameters!(mod_param, 
                                      x_prop, 
                                      param_info, 
                                      i_range_ind, 
                                      i_range_error)             

Mapping proposed parameters (array) to ModelParameters-struct used in particle-filter. 

When proposed via MCMC-samplers the proposed model-parameters (x_prop) are stored in an 
array where individual parameters occupy i_range_ind and error-parameters occupy 
i_range_ind. To be useful in this filter, this array is mapped into the 
mod_param ModelParameters-struct required by the filter. 

See also: [`ModelParameters`](@ref)
"""
function map_proposal_to_model_parameters_ind!(mod_param, 
                                               x_prop, 
                                               param_info, 
                                               i_range_ind, 
                                               i_range_error)             

    # Map the individual rate-parameters 
    j = 1
    for i in i_range_ind
        if param_info.ind_param_log[j] == true  
            mod_param.individual_parameters.c[j] = exp(x_prop[i])
        else
            mod_param.individual_parameters.c[j] = x_prop[i]
        end
        j += 1
    end

    # Map the error parameters 
    j = 1
    for i in i_range_error
        if param_info.error_param_log[j] == true
            mod_param.error_parameters[j] = exp(x_prop[i])
        else
            mod_param.error_parameters[j] = x_prop[i]
        end
        j += 1
    end
end



"""
    write_result_to_file(samples, log_lik_val, param_info, filter, file_loc)

Write posterior samples and log-likelihood values to file for single individual inference. 

Result is stored in dir_save specifiec by file_loc, and param_info object is used
to name parameters in the file. Filter-information is used for tagging teh file-name 
with the number of particles and whether or not correlation is used in the file-name. 
In the future filter-data will be included in the file, rather compared to being 
included in the file-name. 
"""
function write_result_to_file(samples, log_lik_val, param_info, filter, file_loc)
    
    # Process input to get relevant parameters 
    n_ind_param = length(param_info.prior_ind_param) 
    n_error_param = length(param_info.prior_error_param)
    n_param = n_ind_param + n_error_param
    n_particles = filter.n_particles
    n_samples = size(samples)[2]

    run_id = 1
    file_name =  "Npart" * string(n_particles) * 
        "_nsamp" * string(n_samples) * 
        "_corr" * string(filter.ρ) * 
        "_run" * string(run_id) * ".csv"
    path_save = file_loc.dir_save * "/" * file_name

    while true 
        if isfile(path_save)
            run_id += 1
            file_name = "Npart" * string(n_particles) * 
                "_nsamp" * string(n_samples) * 
                "_corr" * string(filter.ρ) *
                "_run" * string(run_id) * ".csv"
            path_save = file_loc.dir_save * "/" * file_name
        else
            break
        end
    end

    col_names = Array{String, 1}(undef, n_param + 1)
    j = 1
    for i in 1:n_ind_param
        col_names[j] = "c" * string(i)
        j += 1
    end
    for i in 1:n_error_param
        col_names[j] = "sigma" * string(i)
        j += 1
    end
    col_names[end] = "log_lik"

    data_write = vcat(samples, log_lik_val')
    df = DataFrame(data_write', :auto)
    rename!(df, col_names)

    # Ensure that dir-saves exists 
    if !isdir(file_loc.dir_save)
        mkpath(file_loc.dir_save)
    end
    CSV.write(path_save, df)

end


"""
    run_mcmc(n_samples, mcmc_sampler, param_info, filter, model, file_loc; pilot=false)

Run pseudo-marginal particle mcmc for single-individual inference. 

For a specific mcmc-sampler (either Random-walk, AM-sampler or general-AM) sample
n_samples from the posterior parameter distribution for provided model using 
inital values, and other parameter characterics, specified by param_info using a 
particle filter specified by filter. It is recomended to initalise the starting 
parameters for param_info (starting values), mcmc-sampler (e.g covariance matrix) 
and filter (number of particles) via a pilot-run. pilot=true is used when 
running sampler in pilot-run setting. 

See also: [`run_pilot_run`](@ref)
"""
function run_mcmc(n_samples, mcmc_sampler, param_info, filter, model::SdeModel, file_loc; pilot=false)

    # Set up correct file-locations
    calc_dir_save!(file_loc, filter, mcmc_sampler)

    # Get indices for proposing new parameters and for mapping 
    # proposed parameters correctly to IndPrameter struct (only compute once)
    n_param_infer = length(param_info.prior_ind_param) + length(param_info.prior_error_param)
    i_range_ind = 1:length(param_info.prior_ind_param)
    i_range_error = length(param_info.prior_ind_param)+1:n_param_infer
    prior_dists = vcat(param_info.prior_ind_param, param_info.prior_error_param)
    positive_proposal = vcat(param_info.ind_param_pos, param_info.error_param_pos)
    n_particles = filter.n_particles

    # Set up the parameter-samples matrix 
    param_sampled = Array{Float64, 2}(undef, (n_param_infer, n_samples))
    param_sampled[:, 1] = vcat(param_info.init_ind_param, param_info.init_error_param)
    x_prop = Array{Float64, 1}(undef, n_param_infer)
    x_old = param_sampled[1:n_param_infer, 1]

    # Compute the individual data 
    data_obs = CSV.read(file_loc.path_data, DataFrame)
    ind_data = init_ind_data(data_obs, filter)

    # Init the model parameters object
    mod_param = init_model_parameters(param_info.init_ind_param, 
                                      param_info.init_error_param, 
                                      model, 
                                      covariates=ind_data.cov_val)

    map_proposal_to_model_parameters_ind!(mod_param, x_old, param_info, 
                                          i_range_ind, i_range_error)

    # Print some data 
    str_write = @sprintf("Running inference with %d particles and correlation level %.3f\n", n_particles, filter.ρ)
    @info "$str_write"

    # Initialise the random numbers 
    rand_num_old = create_rand_num(ind_data, model, filter)

    # Cache with vectors required by particle filter 
    filter_cache = create_cache(filter, Val(model.dim_obs), Val(model.dim), Val(size(model.P_mat)[2]), model.P_mat)

    # Storing the log-likelihood values 
    log_lik_val = Array{Float64, 1}(undef, n_samples)

    # Calculate likelihood, jacobian and prior for initial parameters 
    log_prior_old = calc_log_prior(x_old, prior_dists, n_param_infer)
    log_jacobian_old = calc_log_jac(x_old, positive_proposal, n_param_infer)
    log_lik_old = run_filter(filter, mod_param, rand_num_old, filter_cache, model, ind_data, Val(filter.is_correlated))
    log_lik_val[1] = log_lik_old

    str_write = @sprintf("Log_lik_start = %.3f\n", log_lik_old)
    @info "$str_write"

    # Run mcmc-sampling 
    @showprogress 1 "Running sampler..." for i in 2:n_samples
        # Ensure that correct random-numbers are used for propegating 
        rand_num_new = deepcopy(rand_num_old)

        # Propose new-parameters (when old values are used)
        update_random_numbers!(rand_num_new, filter)
        propose_parameters(x_prop, x_old, mcmc_sampler, 
                        n_param_infer, positive_proposal)
        map_proposal_to_model_parameters_ind!(mod_param, x_prop, param_info, 
                                              i_range_ind, i_range_error)

        # Calculate new jacobian, log-likelihood and prior prob >
        log_prior_new = calc_log_prior(x_prop, prior_dists, n_param_infer)
        log_jacobian_new = calc_log_jac(x_prop, positive_proposal, n_param_infer)

        log_lik_new = run_filter(filter, mod_param, rand_num_new, filter_cache, model, ind_data, Val(filter.is_correlated))

        # Acceptange probability
        log_u = log(rand())
        log_alpha = (log_lik_new - log_lik_old) + (log_prior_new - log_prior_old) + 
            (log_jacobian_old - log_jacobian_new)

        # In case of very bad-parameters (NaN) do not accept 
        if isnan(log_alpha)
            log_alpha = -Inf 
        end

        # Accept 
        if log_u < log_alpha
            log_lik_old = log_lik_new
            log_prior_old = log_prior_new
            log_jacobian_old = log_jacobian_new
            param_sampled[:, i] .= x_prop
            x_old .= x_prop
            rand_num_old = rand_num_new
        # Do not accept 
        else
            param_sampled[:, i] .= x_old
        end

        log_lik_val[i] = log_lik_old

        # Update-adaptive mcmc-sampler 
        update_sampler!(mcmc_sampler, param_sampled, i, log_alpha)
    end

    # If not running pilot, save result 
    if pilot == false
        write_result_to_file(param_sampled, log_lik_val, param_info, filter, 
            file_loc)
    end

    # Perform posterior visual check 
    #=
    if pilot == false
        @printf("Runing posterior visual check...")
        posterior_vc(param_sampled, model, file_loc, param_info, filter, ind_data)
        @printf("done.\n")
    end
    =#

    return param_sampled, log_lik_val, mcmc_sampler
end
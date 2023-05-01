#= 
    Test if the particle filter produces an unbiased likelihood estimate by comparing the particle filter against 
    the Ornstein model for which we can compute the likelihood analytically 
=# 


using Distributions 
using Random 
using LinearAlgebra 
using Plots
using StatsPlots
using Test

# For simulating data 
include(joinpath(@__DIR__, "Simulating_data.jl"))
# For inference
include(joinpath(@__DIR__, "..", "src", "SDE_inference.jl"))
# For Ornstein model 
include(joinpath(@__DIR__, "Ornstein_model.jl"))


# Code taken from Wiqvist, Kalman filter for the Ornstein-Uhlenbeck model 
function kalman_filter(y::Vector{Float64}, σ_ϵ::Float64, log_c::Vector{Float64}, dt::Float64; p_start=0.3^2)::Float64

    T = length(y)

    #println(loglik_est[m])
    θ_1::Float64 = exp(log_c[1])
    θ_2::Float64 = exp(log_c[2])
    θ_3::Float64 = exp(log_c[3])

    # set model
    B::Float64 = θ_1*θ_2
    A::Float64 = -θ_1
    σ::Float64 = θ_3
    C = 1
    S::Float64 = σ_ϵ^2

    # start values for the kalman filter
    P_start = p_start
    x_hat_start = 0.0

    P_k = P_start
    x_k = x_hat_start

    loglik_est::Float64 = 0.0

    # main loop
    for k = 1:T

        x_k = exp(A*dt)*x_k + (1/A)*(exp(dt*A)-1)*B 
        P_k = exp(A*dt)*P_k*exp(A*dt) + σ^2*(exp(2*A*dt)-1) / (2*A)

        R_k = C*P_k*C + S
        K = P_k*C*inv(R_k)
        ϵ_k = y[k]-C*x_k
        x_k = x_k + K*ϵ_k
        P_k = P_k - K*R_k*K

        loglik_est = loglik_est - 0.5*(log(det(R_k)) + ϵ_k*inv(R_k)*ϵ_k)

    end

    return loglik_est
end


function run_kalman(A_mat, C_mat, Q_mat, R_mat, P_mat, y_data)

    # Arrays to store the mean and covariance matrix for X 
    dim_x = length(A_mat)
    dim_y = length(R_mat)
    cov_ret = Array{Float64, 3}(undef, (dim_x, dim_x, T+1))
    mean_ret = Array{Float64, 2}(undef, (dim_x, T+1))

    cov_ret[:, :, 1] .= P_mat
    mean_ret[:, 1] .= 0.0

    # Allocate Kalman-gain matrix 
    mean_vec = Array{Float64, 1}(undef, dim_x)
    mean_vec .= 0.0
    y_curr = Array{Float64, 1}(undef, dim_y)

    
    for i in 2:(T+1)
        # Update p-matrix and Kalman gain 
        cov_time = A_mat*cov_ret[:, :, i-1]*A_mat' + Q_mat

        kalman_gain = cov_time * C_mat' * inv(C_mat * cov_time * C_mat' + R_mat)

        y_curr .= y_data[i-1]

        # Update mean and covariance
        mean_vec .= A_mat*mean_ret[:, i-1] + kalman_gain*(y_curr - C_mat*A_mat*mean_ret[:, i-1])
        P_mat = cov_time - kalman_gain*C_mat*cov_time

        cov_ret[:, :, i] .= P_mat
        mean_ret[:, i] .= mean_vec

    end

    return mean_ret, cov_ret
end


function run_mcmc_kalman(n_samples, mcmc_sampler, _param_info, path_data)

    @info "Testing inference with Kalman filter"

    param_info = deepcopy(_param_info)
    data_obs = CSV.read(path_data, DataFrame)
    y_data, t_vec = data_obs[!, "obs"], data_obs[!, "time"]
    delta_t = t_vec[2] - t_vec[1]
    
    # Information regarding number of parameters to infer 
    n_param_infer = length(param_info.prior_ind_param) + length(param_info.prior_error_param)
    prior_dists = vcat(param_info.prior_ind_param, param_info.prior_error_param)
    positive_proposal = vcat(param_info.ind_param_pos, param_info.error_param_pos)
        
    # Storing the log-likelihood values 
    log_lik_val = Array{Float64, 1}(undef, n_samples)

    # Storing mcmc-chain 
    param_sampled = Array{Float64, 2}(undef, (n_param_infer, n_samples))
    param_sampled[:, 1] = vcat(param_info.init_ind_param, param_info.init_error_param)
    x_prop = Array{Float64, 1}(undef, n_param_infer)
    x_old = param_sampled[1:n_param_infer, 1]

    log_c = Array{Float64, 1}(undef, 3)
    log_c .= x_old[1:3]
    sigma = x_old[4]

    # Calculate likelihood, jacobian and prior for initial parameters 
    log_prior_old = calc_log_prior(x_old, prior_dists, n_param_infer)
    log_jacobian_old = calc_log_jac(x_old, positive_proposal, n_param_infer)
    log_lik_old = kalman_filter(y_data, sigma, log_c, delta_t)
    log_lik_val[1] = log_lik_old

    @showprogress 1 "Running sampler..." for i in 2:n_samples
        
        # Propose new-parameters (when old values are used)
        propose_parameters(x_prop, x_old, mcmc_sampler, n_param_infer, positive_proposal)
        
        log_c .= x_prop[1:3]
        sigma = x_prop[4]
        # Calculate new jacobian, log-likelihood and prior prob >
        log_prior_new = calc_log_prior(x_prop, prior_dists, n_param_infer)
        log_jacobian_new = calc_log_jac(x_prop, positive_proposal, n_param_infer)
        log_lik_new = kalman_filter(y_data, sigma, log_c, delta_t)
        # Acceptange probability
        log_u = log(rand())
        log_alpha = (log_lik_new - log_lik_old) + (log_prior_new - log_prior_old) + (log_jacobian_old - log_jacobian_new)

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
        # Do not accept 
        else
            param_sampled[:, i] .= x_old
        end

        log_lik_val[i] = log_lik_old

        # Update-adaptive mcmc-sampler 
        update_sampler!(mcmc_sampler, param_sampled, i, log_alpha)
    end

    return log_lik_val, param_sampled
end


function run_ornstein_filter(mcmc_sampler, _param_info, path_data, which_filter)
    
    param_info = deepcopy(_param_info)

    file_loc = init_file_loc(path_data, "Ornstein_single", dir_save = joinpath(@__DIR__, "Ornstein_model", "Single_individual"))
    sde_mod = init_sde_model(alpha_ornstein_full, 
                             beta_ornstein_full, 
                             calc_x0_ornstein!, 
                             ornstein_obs, 
                             prob_ornstein_full,
                             1, 1, 
                             ones(Int64, 1, 1))

    if which_filter === :bootstrap
        @info "Testing inference with bootstrap filter"
        dt = 1e-2
        filter_opt = BootstrapFilterEM(dt, 40, correlation=0.99)
    elseif which_filter === :modifed_diffusion_bridge
        @info "Testing inference with modified diffusion bridge filter"
        dt = 1e-2
        filter_opt = ModifedDiffusionBridgeFilter(dt, 10, correlation=0.99)
    end

    samples, log_lik_val, sampler = run_mcmc(60000, mcmc_sampler, param_info, filter_opt, sde_mod, deepcopy(file_loc))
    return samples 
end


Random.seed!(123)
# We initialize from the true MCMC values 
param_info = init_param([Normal(0.0, 1.0), Normal(2.0, 1.0), Normal(-1.0, 1.0)], 
                        [Gamma(1.0, 0.4)], 
                        ind_param_pos=false, 
                        ind_param_log=true,
                        error_param_pos=true, 
                        init_ind_param=[0.1, 2.3, -0.9], 
                        init_error_param=[0.3])
# Well tuned covariance matrix  
cov_mat = [0.0051533758012418905 -0.0005981880918196099 0.009437567242287902 -0.0024174316959147428; -0.0005981880918196099 0.0004487581124181398 -0.001225268353309352 0.0002915292200297846; 0.009437567242287902 -0.001225268353309352 0.2953266899340503 -0.04807565788209376; -0.0024174316959147428 0.0002915292200297846 -0.04807565788209376 0.015104372987308889]
mcmc_sampler = init_mcmc(RandomWalk(), param_info, step_length=1.0, cov_mat=cov_mat)
path_data = joinpath(@__DIR__, "Ornstein_model", "Simulated_data.csv")

log_lik_kalman, samples_kalman = run_mcmc_kalman(600000, mcmc_sampler, param_info, path_data)
samples_bootstrap = run_ornstein_filter(mcmc_sampler, param_info, path_data, :bootstrap)
samples_bridge = run_ornstein_filter(mcmc_sampler, param_info, path_data, :modifed_diffusion_bridge)

burn_in = 20000
median_kalman = median(samples_kalman[:, burn_in:end], dims=2)
median_bootstrap = median(samples_bootstrap[:, burn_in:end], dims=2)
median_bridge = median(samples_bridge[:, burn_in:end], dims=2)
sd_kalman = std(samples_kalman[:, burn_in:end], dims=2)
sd_bootstrap = std(samples_bootstrap[:, burn_in:end], dims=2)
sd_bridge = std(samples_bridge[:, burn_in:end], dims=2)

@testset "Testing particle filters against Ornsten model" begin
    @test all(abs.(median_kalman .- median_bootstrap) .≤ 0.08)
    @test all(abs.(median_kalman .- median_bridge) .≤ 0.08)
    @test all(abs.(sd_kalman .- sd_bootstrap) .≤ 0.08)
    @test all(abs.(sd_kalman .- sd_bridge) .≤ 0.08)
end

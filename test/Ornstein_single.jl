using Distributions 
using Random 
using LinearAlgebra 
using Plots
using StatsPlots

# For simulating data 
include(joinpath(@__DIR__, "Simulating_data.jl"))
# For inference
include(joinpath(@__DIR__, "..", "src", "SDE_inference.jl"))

# Define necessary functions for Ornstein model 
function alpha_ornstein_full(du, u, p, t)
    c = p.c
    du[1] = c[1] * (c[2] - u[1])
end
function beta_ornstein_full(du, u, p, t)
    c = p.c
    du[1, 1] = c[3]^2
end
function prob_ornstein_full(y_obs, y_mod, error_param, t, dim_obs)

    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)

    # Log-scale for numerical accuracy
    prob = exp(prob)

    return prob
end
function ornstein_obs(y, u, p, t)
    y[1] = u[1]
end
function calc_x0_ornstein!(x0, ind_param)
    x0[1] = 0.0
end
function empty_func() # Can be useful to have 
end


# Setting up SDE-model and simulating data 
Random.seed!(123)
sde_mod = init_sde_model(alpha_ornstein_full, 
                         beta_ornstein_full, 
                         calc_x0_ornstein!, 
                         ornstein_obs, 
                         prob_ornstein_full,
                         1, 1)

error_dist = Normal(0.0, 0.3)
c = exp.([-0.7, 2.3, -0.9])
p = DynModInput(c, [0.0], [0.0])
x0 = [0.0]; t_vec = 0.25:0.25:10
t_vec, y_vec = simulate_data_sde(sde_mod, error_dist, t_vec, p, x0, dt=1e-5)
sim_result = [SimResult(t_vec, y_vec, 1, sde_mod.dim_obs)]
# Writing result to file
write_simulated_data(sim_result, joinpath(@__DIR__, "Ornstein_model"), "Simulated_data.csv")

# Set up inference options 
prior_ind_param = [Normal(0.0, 1.0), Normal(1.0, 1.0), Normal(0.0, 1.0)]
prior_error_param = [Gamma(1.0, 0.4)]
param_info = init_param(prior_ind_param, 
                        prior_error_param, 
                        ind_param_pos=false, 
                        ind_param_log=true,
                        error_param_pos=true)

# Filter information 
dt = 1e-2
rho = 0.99 # Correlation level 
filter_opt = init_filter(BootstrapEm(), dt, rho=rho)

# Set file paths 
path_data = joinpath(@__DIR__, "Ornstein_model", "Simulated_data.csv")
file_loc = init_file_loc(path_data, "Ornstein_single", dir_save = joinpath(@__DIR__, "Ornstein_model", "Single_individual"))

# For Gibbs-step 1&2 adaptive mcmc
cov_mat = diagm([0.1, 0.1, 0.1, 0.1])
mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat)

# Options for pilot run
Random.seed!(123)
pilot_run_info = init_pilot_run([prior_ind_param, prior_error_param], 
                                 n_particles_pilot=1000, 
                                 n_samples_pilot=3000,
                                 n_particles_investigate=[20, 50],
                                 n_times_run_filter=200,
                                 rho_list=[0.99])
tune_particles_single_individual(pilot_run_info, mcmc_sampler, param_info, filter_opt, sde_mod, deepcopy(file_loc))

# Main inference run 
# Get values from pilot run 
Random.seed!(123)
param_info_new = change_start_val_to_pilots(param_info, file_loc, filter_opt, sampler_name = "Ram_sampler")
mcmc_sampler = init_mcmc_pilot(mcmc_sampler, file_loc, filter_opt.rho)
# Actual main run 
filter_opt = init_filter(BootstrapEm(), dt, rho=rho, n_particles=10) # From pilot run use 10 particles 
samples, log_lik_val, sampler = run_mcmc(50000, mcmc_sampler, param_info_new, filter_opt, sde_mod, deepcopy(file_loc))   

burn_in = 20000
p1 = density(samples[1, burn_in:end])
p1 = vline!([-0.7])
p2 = density(samples[2, burn_in:end])
p2 = vline!([2.3])
p3 = density(samples[3, burn_in:end])
p3 = vline!([-0.9])

# Letus try with modified diffusion bridge (here we need special form on SDE-model)
Random.seed!(123)
P = ones(Int64, 1, 1)
sde_mod = init_sde_model(alpha_ornstein_full, 
                         beta_ornstein_full, 
                         calc_x0_ornstein!, 
                         ornstein_obs, 
                         prob_ornstein_full,
                         1, 1, 
                         P)
dt = 1e-2
rho = 0.99 # Correlation level 
filter_opt = init_filter(ModDiffusion(), dt, rho=rho)

# Set file paths 
path_data = joinpath(@__DIR__, "Ornstein_model", "Simulated_data.csv")
file_loc = init_file_loc(path_data, "Ornstein_single", dir_save = joinpath(@__DIR__, "Ornstein_model", "Single_individual_mod_bridge"))

# For Gibbs-step 1&2 adaptive mcmc
cov_mat = diagm([0.1, 0.1, 0.1, 0.1])
mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat)

# Options for pilot run
Random.seed!(123)
pilot_run_info = init_pilot_run([prior_ind_param, prior_error_param], 
                                 n_particles_pilot=100, 
                                 n_samples_pilot=3000,
                                 n_particles_investigate=[20, 50],
                                 n_times_run_filter=200,
                                 rho_list=[0.99])
tune_particles_single_individual(pilot_run_info, mcmc_sampler, param_info, filter_opt, sde_mod, deepcopy(file_loc))

# Main inference run 
# Get values from pilot run 
Random.seed!(123)
param_info_new = change_start_val_to_pilots(param_info, file_loc, filter_opt, sampler_name = "Ram_sampler")
mcmc_sampler = init_mcmc_pilot(mcmc_sampler, file_loc, filter_opt.rho)
# Actual main run 
filter_opt = init_filter(ModDiffusion(), dt, rho=rho, n_particles=10) # From pilot run use 10 particles 
samples, log_lik_val, sampler = run_mcmc(50000, mcmc_sampler, param_info_new, filter_opt, sde_mod, deepcopy(file_loc))  

burn_in = 20000
p1 = density(samples[1, burn_in:end])
p1 = vline!([-0.7])
p2 = density(samples[2, burn_in:end])
p2 = vline!([2.3])
p3 = density(samples[3, burn_in:end])
p3 = vline!([-0.9])
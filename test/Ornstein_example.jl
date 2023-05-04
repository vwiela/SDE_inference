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
                         1, 1, ones(Int64, 1, 1))

# Set up inference options 
param_info = init_param([Normal(0.0, 1.0), Normal(2.0, 1.0), Normal(-1.0, 1.0)], 
                        [Gamma(1.0, 0.4)], 
                        ind_param_pos=false, 
                        ind_param_log=true,
                        error_param_pos=true)
# Filter information 
dt = 1e-2
filter_opt = BootstrapFilterEM(dt, 40, correlation=0.99)

# Set file paths 
path_data = joinpath(@__DIR__, "Ornstein_model", "Simulated_data.csv")
file_loc = init_file_loc(path_data, "Ornstein_single", dir_save = joinpath(@__DIR__, "Ornstein_model", "Single_individual"))

# For Gibbs-step 1&2 adaptive mcmc
cov_mat = diagm([0.1, 0.1, 0.1, 0.1])
mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat)

# Options for pilot run
Random.seed!(123)
pilot_run_info = init_pilot_run([[Normal(0.0, 1.0), Normal(2.0, 1.0), Normal(-1.0, 1.0)], [Gamma(1.0, 0.4)]], 
                                 n_particles_pilot=1000, 
                                 n_samples_pilot=3000,
                                 n_particles_investigate=[20, 50],
                                 n_times_run_filter=200,
                                 ρ_list=[0.99])
tune_particles_single_individual(pilot_run_info, mcmc_sampler, param_info, filter_opt, sde_mod, deepcopy(file_loc))

# Main inference run 
# Get values from pilot run 
Random.seed!(123)
param_info_new = change_start_val_to_pilots(param_info, file_loc, filter_opt, sampler_name = "Ram_sampler")
mcmc_sampler = init_mcmc_pilot(mcmc_sampler, file_loc, filter_opt.ρ)
# Actual main run 
filter_opt = remake_filter(filter_opt, n_particles=20) # From pilot run use 10 particles 
samples, log_lik_val, sampler = run_mcmc(50000, mcmc_sampler, param_info_new, filter_opt, sde_mod, deepcopy(file_loc))   

burn_in = 20000
p1 = density(samples[1, burn_in:end])
p1 = vline!([0.1])
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
filter_opt = ModifedDiffusionBridgeFilter(dt, 100, correlation=rho)

# Set file paths 
path_data = joinpath(@__DIR__, "Ornstein_model", "Simulated_data.csv")
file_loc = init_file_loc(path_data, "Ornstein_single", dir_save = joinpath(@__DIR__, "Ornstein_model", "Single_individual_mod_bridge"))

# For Gibbs-step 1&2 adaptive mcmc
cov_mat = diagm([0.1, 0.1, 0.1, 0.1])
mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat)

# Options for pilot run
Random.seed!(123)
param_info = init_param([Normal(0.0, 1.0), Normal(2.0, 1.0), Normal(-1.0, 1.0)], 
                        [Gamma(1.0, 0.4)], 
                        ind_param_pos=false, 
                        ind_param_log=true,
                        error_param_pos=true)
tune_particles_single_individual(pilot_run_info, mcmc_sampler, param_info, filter_opt, sde_mod, deepcopy(file_loc))

# Main inference run 
# Get values from pilot run 
Random.seed!(123)
param_info_new = change_start_val_to_pilots(param_info, file_loc, filter_opt, sampler_name = "Ram_sampler")
mcmc_sampler = init_mcmc_pilot(mcmc_sampler, file_loc, rho)
# Actual main run 
dt = 1e-2
filter_opt = ModifedDiffusionBridgeFilter(dt, 10)
samples, log_lik_val, sampler = run_mcmc(50000, mcmc_sampler, param_info_new, filter_opt, sde_mod, deepcopy(file_loc))  

burn_in = 20000
p1 = density(samples[1, burn_in:end])
p1 = vline!([0.1])
p2 = density(samples[2, burn_in:end])
p2 = vline!([2.3])
p3 = density(samples[3, burn_in:end])
p3 = vline!([-0.9])


#=
dt=1e-2
filter_opt = BootstrapFilterEM(dt, 40, correlation=0.99)
filter_cache = create_cache(filter_opt, Val(sde_mod.dim_obs), Val(sde_mod.dim), Val(1), sde_mod.P_mat)
path_data = joinpath(@__DIR__, "Ornstein_model", "Simulated_data.csv")
ind_data = init_ind_data(CSV.read(path_data, DataFrame), filter_opt)
mod_param = ModelParameters(DynModInput(exp.([0.1, 2.3, -0.9]), Float64[], Float64[]), Float64[], [0.3], Float64[])
random_numbers = create_rand_num(ind_data, sde_mod, filter_opt)
log_lik = run_filter(filter_opt, mod_param, random_numbers, filter_cache, sde_mod, ind_data, Val(filter_opt.is_correlated))
@allocated log_lik = run_filter(filter_opt, mod_param, random_numbers, filter_cache, sde_mod, ind_data, Val(filter_opt.is_correlated))
bTime =  @elapsed log_lik = run_filter(filter_opt, mod_param, random_numbers, filter_cache, sde_mod, ind_data)
=#

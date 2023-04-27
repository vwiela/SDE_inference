
  
using Distributions
using CSV
using DataFrames
using Printf
using Random
using LinearAlgebra


# Struct contianing simulation result for an individual.
# Args:
#     t_vec, time-vector
#     y_vec, y-matrix (each row is an observation) for the observed
#       time-points
#     id, id of the simulated individual
struct SimResult
    t_vec::Array{Float64, 1}
    y_vec::Array{Float64, 2}
    id::Int64
    dim_obs::Int64
end


# Function that writes simulated data to file for one or more individuals.
# The data is stored in a tidy-format, that is each row corresponds to one
# observation for one time-point. The data is also saved with an id referring
# to the individual which the data corresponds to.
# Args:
#     sim_result, array of sim_result entries
#     dir_save, directory where to save the result
#     file_name, name of file to be saved
#     dim_obs, dimension of observeble
# Returns:
#     void
function write_simulated_data(sim_result, dir_save, file_name)

    # Sanity check input
    y_vec = sim_result[1].y_vec
    dim_obs = sim_result[1].dim_obs
    if size(y_vec)[1] != dim_obs
        print("Error, y_vec has wrong dimensions. Should have equal number
        of rows as observed outputs\n")
    end
    if !isa(sim_result, Array)
        print("Error, sim_result must be an array")
    end

    # Ensure directory exists
    mkpath(dir_save)
    file_save = joinpath(dir_save, file_name)
    # If file exists, delete as appending later
    if isfile(file_save)
        rm(file_save)
    end

    # Write each output variable for all individuals
    # i ensures correctly written header
    i = 1
    for sim_res in sim_result
        for j in 1:dim_obs
            df = DataFrame()
            lab = string(j)
            len = length(sim_res.t_vec)
            df[!, "time"] = sim_res.t_vec
            df[!, "obs"] = sim_res.y_vec[j, :]
            df[!, "obs_id"] = repeat([lab], len)
            df[!, "id"] = repeat([sim_res.id], len)

            if i == 1 && j == 1
                CSV.write(file_save, df, append=false)
            else
                CSV.write(file_save, df, append=true)
            end
        end
        i += 1
    end

end

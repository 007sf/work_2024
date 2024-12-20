using FFTW
using Plots
using Random

grid_size = 7
sensor_number = 49
second_maxima=Vector{Float64}(undef, sensor_number)  
SMS = Vector{Float64}(undef, 490)  

for ii = 1:10
    for i = 1:sensor_number
        grid_data = zeros(grid_size, grid_size)
        selected_positions = randperm(grid_size^2)[1:i]

        for pos in selected_positions
            grid_data[pos] = 1
        end

        fft_data = fft(grid_data)
        fft_shifted = fftshift(fft_data)
        magnitude = abs.(fft_shifted)
        normalized_magnitude = (magnitude .- minimum(magnitude)) / (maximum(magnitude) - minimum(magnitude))

        sorted_magnitude = sort(vec(magnitude), rev=true)
        second_maxima[i] = sorted_magnitude[2]
        
    end
    SMS[ii] = second_maxima
end

# Define ticks for the x and y axes
x_ticks = 1:grid_size
y_ticks = 1:grid_size
# Plot the magnitude data with no legend
plot1 = heatmap(normalized_magnitude, color=:grays, colorbar=true, legend=false, aspect_ratio=:equal,
               xlims=(0.5, 7.5), ylims=(0.5, 7.5))
savefig(plot1, "fourier_spectrum_$sensor_number sensor.png")


plot2=plot(vec(magnitude),legend=false)
savefig(plot2, "fourier_magnitude_$sensor_number sensor.pdf")


plot3 = heatmap(grid_data, color=:Greys, aspect_ratio=:equal,
                xlims=(0.5, 7.5), ylims=(0.5, 7.5),
                colorbar=false, legend=false, framestyle=:box,
                xticks=1:grid_size, yticks=1:grid_size)

savefig(plot3, "grid_7_$sensor_number sensor.png")


plot4=plot(second_maxima,legend=false)
savefig(plot4, "second_maxima_$sensor_number sensor.png")

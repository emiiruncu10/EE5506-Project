close all; clear all; clc;
% System Parameters
dt = 0.0125;
g = 9.8;
q_c = 1;
r = 0.3;
num_steps = 400;
N_particles = 1000;
N_trials = 100;
threshold = 0.015;
initial_state = [pi / 4; 0];
Q = [q_c * dt^3 / 3, q_c * dt^2 / 2; q_c * dt^2 / 2, q_c * dt];
R = r^2;
state_dim = 2;
% Dynamics and Observation Models
dynamics = @(x) [x(1) + x(2) * dt; x(2) - g * sin(x(1)) * dt];
emission = @(x) sin(x(1));
likelihood = @(y, x) exp(-0.5 * (y - emission(x))^2 / R);

rmse_list = [];
% Ground Truth Generation
for trial = 1: N_trials
    states = zeros(state_dim, num_steps);
    observations = zeros(1, num_steps);
    x = initial_state;
    for k = 1:num_steps
        x = dynamics(x) + mvnrnd([0; 0], Q)';
        y = emission(x) + sqrt(R) * randn();
        states(:, k) = x;
        observations(k) = y;
    end
    time_grid = 0:dt:(num_steps - 1) * dt;
    
    % Particle Filter Initialization
    particles = mvnrnd(initial_state, eye(state_dim)*0.1, N_particles)';
    weights = ones(N_particles, 1) / N_particles;
    estimate = zeros(state_dim, num_steps);
    estimate(:,1) = mean(particles, 2);
    
    % Particle Filter Loop
    for k = 2:num_steps
        % Prediction
        for i = 1:N_particles
            particles(:, i) = dynamics(particles(:, i)) + mvnrnd([0; 0], Q)';
        end
    
        % Weight Update
        for i = 1:N_particles
            weights(i) = weights(i) * likelihood(observations(k), particles(:, i));
        end
        weights = weights / sum(weights);  % Normalize
    
        % Residual Resampling
        indices = residual_resample(weights);
        particles = particles(:, indices);
        weights = ones(N_particles, 1) / N_particles;
    
        % Estimation
        estimate(:, k) = mean(particles, 2);
    end
    
    % RMSE Calculation
    rmse = sqrt(mean((estimate(1,:) - states(1,:)).^2));
    fprintf('[INFO] Particle Filter RMSE: %.4f\n', rmse);
    rmse_list(trial) = rmse;
    
    if (trial == N_trials || trial == 1)
    % Plot
        figure;
        subplot(2,1,1)
        plot(time_grid, states(1,:), 'k-', 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        xlabel('Time t'); ylabel('Pendulum angle x_1,k'); title('Ground Truth & Measurements');
        legend('True Angle', 'Measurements'); grid on;
        
        subplot(2,1,2)
        plot(time_grid, states(1,:), 'k-', 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        plot(time_grid, estimate(1,:), 'b-', 'LineWidth', 2);
        xlabel('Time t'); ylabel('Estimate'); title('Particle Filter Estimate');
        legend('True Angle', 'Measurements', 'PF Estimate'); grid on;
    end
end

avg_rmse = mean(rmse_list);
std_rmse = std(rmse_list);

fprintf("Average RMSE after monte-carlo sim PF: %.2f\n", avg_rmse);
fprintf("Average std RMSE after monte-carlo sim PF: %.2f\n", std_rmse); 

function indices = residual_resample(weights)
% Residual Resampling Algorithm
N = length(weights);
indices = zeros(N, 1);
num_copies = floor(N * weights);
k = 1;

for i = 1:N
    for j = 1:num_copies(i)
        indices(k) = i;
        k = k + 1;
    end
end

residual = N - sum(num_copies);
if residual > 0
    residual_weights = (N * weights - num_copies);
    residual_weights = residual_weights / sum(residual_weights);
    cumsum_weights = cumsum(residual_weights);
    u = rand(residual, 1);
    for j = 1:residual
        indices(k) = find(cumsum_weights >= u(j), 1);
        k = k + 1;
    end
end

indices = indices(randperm(N));  % Shuffle
end
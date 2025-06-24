close all; clear all;
% Parameters
dt = 0.0125;
g = 9.8;
q_c = 1;
r = 0.3;
num_steps = 400;
N_trials = 100;
state_dim = 2;
observation_dim = 1;

initial_state = [pi / 4; 0];
Q = [q_c * dt^3 / 3, q_c * dt^2 / 2; q_c * dt^2 / 2, q_c * dt];
R = r^2;

dynamics = @(x) [x(1) + x(2) * dt; x(2) - g * sin(x(1)) * dt];
emission = @(x) sin(x(1));
% PF parameters
N_particles = 1000;
likelihood = @(y, x) exp(-0.5 * (y - emission(x))^2 / R);
rmse_list_pf = [];
% UKF parameters
alpha = 1e-3; beta = 2; kappa = 0;
n = state_dim;
rmse_list_ukf = [];
% EKF parameters
rmse_list_ekf = [];
time_grid = 0:dt:(num_steps - 1) * dt;

for trial = 1:N_trials
    states = zeros(state_dim, num_steps);
    observations = zeros(observation_dim, num_steps);
    x = initial_state;
    
    for k = 1:num_steps
        x = dynamics(x) + mvnrnd([0; 0], Q)';
        y = emission(x) + sqrt(R) * randn();
        states(:, k) = x;
        observations(:, k) = y;
    end
    % Perform EKF
    % Initial conditions:
    m_ekf(:,1) = initial_state;
    P_ekf(:, :, 1) = eye(state_dim) * 0.1;
        
    % EKF Loop:
    for k = 2:num_steps
        % Prediction
        m_pred_ekf = dynamics(m_ekf(:, k-1));
        F_ekf = [1, dt; -g*cos(m_ekf(1, k-1)) *dt, 1];
        P_pred_ekf = F_ekf * P_ekf(:, :, k-1) * F_ekf' + Q;
        % Update
        H_ekf = [cos(m_pred_ekf(1)), 0];
        y_pred_ekf = emission(m_pred_ekf);
        S = H_ekf * P_pred_ekf * H_ekf' + R;
        K_ekf = (P_pred_ekf * H_ekf')/ S;
        m_ekf(:, k) = m_pred_ekf + K_ekf* (observations(k) - y_pred_ekf);
        P_ekf(:, :, k) = (eye(state_dim) - K_ekf*H_ekf) * P_pred_ekf;
    end

    rmse_val_ekf = compute_rmse(states(1, :)', m_ekf(1, :)');
    rmse_list_ekf(trial) = rmse_val_ekf;
    fprintf('RMSE Val for trial(%d) EKF = %.2f\n', trial, rmse_val_ekf);
    
    % Plot Ekf
    if (trial == 1)
        figure(1);
        subplot(2,1,1)
        plot(time_grid, states(1, :), 'Color', [0.3,0.3,0.3], 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        title('Data Sample')
        xlabel('Time t');               
        ylabel('Pendulum angle x_1,k'); 
        xlim([0 5]);
        ylim([-3 5]);
        legend('True Angle', 'Measurements');
        grid on;
        subplot(2,1,2)
        plot(time_grid, states(1, :), 'Color', [0.3,0.3,0.3], 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        plot(time_grid, m_ekf(1, :), 'Color', [0,0.7,0.9], 'LineWidth', 3.0);
        title('EKF Estimate First Epoch')
        xlabel('Time t');               
        ylabel('Pendulum angle x_1,k'); 
        xlim([0 5]);
        ylim([-3 5]);
        legend('True Angle', 'Measurements', 'EKF Estimate');
        grid on;
    end

    % Perform UKF
    % UKF storage
    m_ukf = zeros(state_dim, num_steps);
    P_ukf = zeros(state_dim, state_dim, num_steps);
    m_ukf(:, 1) = initial_state;
    P_ukf(:, :, 1) = eye(state_dim) * 0.1;
    
    % UKF loop
    for k = 2:num_steps
        % Generate sigma points
        [X, Wm, Wc] = sigma_points(m_ukf(:, k-1), P_ukf(:, :, k-1), alpha, beta, kappa);
        
        % Predict step
        X_pred_ukf = zeros(state_dim, 2*n + 1);
        for i = 1:(2*n + 1)
            X_pred_ukf(:, i) = dynamics(X(:, i));
        end
        m_pred_ukf = X_pred_ukf * Wm;
        P_pred = Q;
        for i = 1:(2*n + 1)
            dx = X_pred_ukf(:, i) - m_pred_ukf;
            P_pred = P_pred + Wc(i) * (dx * dx');
        end
    
        % Observation prediction
        Y = zeros(observation_dim, 2*n + 1);
        for i = 1:(2*n + 1)
            Y(:, i) = emission(X_pred_ukf(:, i));
        end
        y_pred_ukf = Y * Wm;
        Pyy = R;
        for i = 1:(2*n + 1)
            dy = Y(:, i) - y_pred_ukf;
            Pyy = Pyy + Wc(i) * (dy * dy');
        end
    
        % Cross covariance
        Pxy = zeros(state_dim, observation_dim);
        for i = 1:(2*n + 1)
            dx = X_pred_ukf(:, i) - m_pred_ukf;
            dy = Y(:, i) - y_pred_ukf;
            Pxy = Pxy + Wc(i) * (dx * dy');
        end
    
        % Kalman gain and update
        K = Pxy / Pyy;
        m_ukf(:, k) = m_pred_ukf + K * (observations(:, k) - y_pred_ukf);
        P_ukf(:, :, k) = P_pred - K * Pyy * K';
    end
    
    rmse_val_ukf = compute_rmse(states(1, :)', m_ukf(1, :)');
    rmse_list_ukf(trial) = rmse_val_ukf;
    fprintf('RMSE Val for trial(%d) UKF = %.2f\n', trial, rmse_val_ukf);
    
    if (trial == 1)
        figure(2);
        subplot(2,1,1)
        plot(time_grid, states(1, :), 'Color', [0.3, 0.3, 0.3], 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        title('UKF - Data Sample')
        xlabel('Time t'); ylabel('Pendulum angle x_1,k');
        xlim([0 5]); ylim([-3 5]);
        legend('True Angle', 'Measurements');
        grid on;
        
        subplot(2,1,2)
        plot(time_grid, states(1, :), 'Color', [0.3, 0.3, 0.3], 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        plot(time_grid, m_ukf(1, :), 'Color', [0, 0.7, 0.9], 'LineWidth', 3.0);
        title('UKF Estimate First Epoch')
        xlabel('Time t'); ylabel('Pendulum angle x_1,k');
        xlim([0 5]); ylim([-3 5]);
        legend('True Angle', 'Measurements', 'UKF Estimate');
        grid on;
    end

    % Perform PF
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
    rmse_val_pf = sqrt(mean((estimate(1,:) - states(1,:)).^2));
    rmse_list_pf(trial) = rmse_val_pf;
    fprintf('RMSE Val for trial(%d) PF = %.2f\n', trial,rmse_val_pf);
    
    if (trial == 1)
        figure(3);
        subplot(2,1,1)
        plot(time_grid, states(1,:), 'k-', 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        xlabel('Time t'); ylabel('Pendulum angle x_1,k'); title('Ground Truth & Measurements');
        legend('True Angle', 'Measurements'); grid on;
        title('PF Estimate First Epoch')
        subplot(2,1,2)
        plot(time_grid, states(1,:), 'k-', 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        plot(time_grid, estimate(1,:), 'b-', 'LineWidth', 2);
        xlabel('Time t'); ylabel('Estimate'); title('Particle Filter Estimate');
        legend('True Angle', 'Measurements', 'PF Estimate'); grid on;
    end
    
    if (trial == N_trials)
        % plot EKF last Epoch
        figure(4);
        subplot(2,1,1)
        plot(time_grid, states(1, :), 'Color', [0.3,0.3,0.3], 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        title('Data Sample')
        xlabel('Time t');               
        ylabel('Pendulum angle x_1,k'); 
        xlim([0 5]);
        ylim([-3 5]);
        legend('True Angle', 'Measurements');
        grid on;
        subplot(2,1,2)
        plot(time_grid, states(1, :), 'Color', [0.3,0.3,0.3], 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        plot(time_grid, m_ekf(1, :), 'Color', [0,0.7,0.9], 'LineWidth', 3.0);
        title('EKF Estimate Last Epoch')
        xlabel('Time t');               
        ylabel('Pendulum angle x_1,k'); 
        xlim([0 5]);
        ylim([-3 5]);
        legend('True Angle', 'Measurements', 'EKF Estimate');
        grid on;
        % plot UKF last Epoch
        figure(5);
        subplot(2,1,1)
        plot(time_grid, states(1, :), 'Color', [0.3, 0.3, 0.3], 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        title('UKF - Data Sample')
        xlabel('Time t'); ylabel('Pendulum angle x_1,k');
        xlim([0 5]); ylim([-3 5]);
        legend('True Angle', 'Measurements');
        grid on;
        subplot(2,1,2)
        plot(time_grid, states(1, :), 'Color', [0.3, 0.3, 0.3], 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        plot(time_grid, m_ukf(1, :), 'Color', [0, 0.7, 0.9], 'LineWidth', 3.0);
        title('UKF Estimate Last Epoch')
        xlabel('Time t'); ylabel('Pendulum angle x_1,k');
        xlim([0 5]); ylim([-3 5]);
        legend('True Angle', 'Measurements', 'UKF Estimate');
        grid on;
        % plot PF last Epoch
        figure(6)
        subplot(2,1,1)
        plot(time_grid, states(1,:), 'k-', 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        xlabel('Time t'); ylabel('Pendulum angle x_1,k'); title('Ground Truth & Measurements');
        legend('True Angle', 'Measurements'); grid on;
        title('PF Estimate Last Epoch')
        subplot(2,1,2)
        plot(time_grid, states(1,:), 'k-', 'LineWidth', 2.5); hold on;
        plot(time_grid, observations, 'ko', 'MarkerSize', 2);
        plot(time_grid, estimate(1,:), 'b-', 'LineWidth', 2);
        xlabel('Time t'); ylabel('Estimate'); title('Particle Filter Estimate');
        legend('True Angle', 'Measurements', 'PF Estimate'); grid on;
    end
end

avg_rmse_ekf = mean(rmse_list_ekf);
var_rmse_ekf = var(rmse_list_ekf);
avg_rmse_ukf = mean(rmse_list_ukf);
var_rmse_ukf = var(rmse_list_ukf);
avg_rmse_pf  = mean(rmse_list_pf);
var_rmse_pf  = var(rmse_list_pf);

fprintf("Average RMSE after monte-carlo sim EKF: %.4f\n", avg_rmse_ekf);
fprintf("Var RMSE after monte-carlo sim EKF: %.4f\n", var_rmse_ekf); 
fprintf("Average RMSE after monte-carlo sim UKF: %.4f\n", avg_rmse_ukf);
fprintf("Var RMSE after monte-carlo sim UKF: %.4f\n", var_rmse_ukf);
fprintf("Average RMSE after monte-carlo sim PF: %.4f\n", avg_rmse_pf);
fprintf("Var RMSE after monte-carlo sim PF: %.4f\n", var_rmse_pf);

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

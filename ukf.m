close all; clear all; clc;
% Parameters
dt = 0.0125;
g = 9.8;
q_c = 1;
r = 0.3;
num_steps = 400;
state_dim = 2;
observation_dim = 1;
N_trials = 100;

initial_state = [pi / 4; 0];

Q = [q_c * dt^3 / 3, q_c * dt^2 / 2; q_c * dt^2 / 2, q_c * dt];
R = r^2;
dynamics = @(x) [x(1) + x(2) * dt; x(2) - g * sin(x(1)) * dt];
emission = @(x) sin(x(1));

rmse_list = [];
% UKF parameters
alpha = 1e-3; beta = 2; kappa = 0;
n = state_dim;
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
        X_pred = zeros(state_dim, 2*n + 1);
        for i = 1:(2*n + 1)
            X_pred(:, i) = dynamics(X(:, i));
        end
        m_pred = X_pred * Wm;
        P_pred = Q;
        for i = 1:(2*n + 1)
            dx = X_pred(:, i) - m_pred;
            P_pred = P_pred + Wc(i) * (dx * dx');
        end
    
        % Observation prediction
        Y = zeros(observation_dim, 2*n + 1);
        for i = 1:(2*n + 1)
            Y(:, i) = emission(X_pred(:, i));
        end
        y_pred = Y * Wm;
        Pyy = R;
        for i = 1:(2*n + 1)
            dy = Y(:, i) - y_pred;
            Pyy = Pyy + Wc(i) * (dy * dy');
        end
    
        % Cross covariance
        Pxy = zeros(state_dim, observation_dim);
        for i = 1:(2*n + 1)
            dx = X_pred(:, i) - m_pred;
            dy = Y(:, i) - y_pred;
            Pxy = Pxy + Wc(i) * (dx * dy');
        end
    
        % Kalman gain and update
        K = Pxy / Pyy;
        m_ukf(:, k) = m_pred + K * (observations(:, k) - y_pred);
        P_ukf(:, :, k) = P_pred - K * Pyy * K';
    end
    rmse_val = compute_rmse(states(1, :)', m_ukf(1, :)', R, 'UKF');
    fprintf('RMSE Val for trial(%d) = %.2f\n', trial, rmse_val);
    rmse_list(trial) = rmse_val;

    if  (trial == N_trials ||trial == 1)
        % Plot
        figure;
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
        title('UKF Estimate')
        xlabel('Time t'); ylabel('Pendulum angle x_1,k');
        xlim([0 5]); ylim([-3 5]);
        legend('True Angle', 'Measurements', 'UKF Estimate');
        grid on;
    end
end

avg_rmse = mean(rmse_list);
std_rmse = std(rmse_list);

fprintf("Average RMSE after monte-carlo sim UKF: %.2f\n", avg_rmse);
fprintf("Average std RMSE after monte-carlo sim UKF: %.2f\n", std_rmse); 


close all; clear all; clc;
dt = 0.0125;
g = 9.8;
q_c = 1;   %% Spectral-Density of the cont-time noise process
r = 0.3;
num_steps = 400;
state_dim = 2;
observation_dim = 1;

N_trials = 100;
initial_state = [pi / 4; 0];

Q = [q_c * dt^3 / 3, q_c * dt^2 / 2;
     q_c * dt^2 / 2, q_c * dt];

R = r^2;

dynamics = @(x) [x(1) + x(2) * dt; x(2) - g * sin(x(1)) * dt];
emission = @(x) sin(x(1));

rmse_list = [];

for trial = 1: N_trials
    states = zeros(2, num_steps);
    observations = zeros(1, num_steps);
    x = initial_state;
    
    for k = 1:num_steps
        x = dynamics(x) + mvnrnd([0; 0], Q)';
        y = emission(x) + sqrt(R) * randn();
        states(:, k) = x;
        observations(:, k) = y;
    end
    
    time_grid = 0:dt:(num_steps - 1) * dt;
    
    % Initial conditions:
    m_ekf(:,1) = initial_state;
    P_ekf(:, :, 1) = eye(state_dim) * 0.1;
    
    % EKF Loop:
    for k = 2:num_steps
        % Prediction
        m_pred = dynamics(m_ekf(:, k-1));
        F = [1, dt; -g*cos(m_ekf(1, k-1)) *dt, 1];
        P_pred = F * P_ekf(:, :, k-1) * F' + Q;
        % Update
        H = [cos(m_pred(1)), 0];
        y_pred = emission(m_pred);
        S = H * P_pred * H' + R;
        K = (P_pred * H')/ S;
        m_ekf(:, k) = m_pred + K* (observations(k) - y_pred);
        P_ekf(:, :, k) = (eye(state_dim) - K*H) * P_pred;
    end
    
    rmse_val = compute_rmse(states(1, :)', m_ekf(1, :)', R, 'EKF');
    fprintf('RMSE Val for trial(%d) = %.2f\n', trial, rmse_val);
    rmse_list(trial) = rmse_val;
    
    if  (trial == N_trials || trial == 1)
        fprintf('[INFO] Monte Carlo Sim Last Epoch\n');
        %fprintf('Calculated average RMSE: %.2f\n', avg_rmse);
        %fprintf('Calculated average std_rmse: %.2f\n', std_rmse);
        % Plot The Latest measurement
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
        title('EKF Estimate')
        xlabel('Time t');               
        ylabel('Pendulum angle x_1,k'); 
        xlim([0 5]);
        ylim([-3 5]);
        legend('True Angle', 'Measurements', 'EKF Estimate');
        grid on;
    end
end

avg_rmse = mean(rmse_list);
std_rmse = std(rmse_list);

fprintf("Average RMSE after monte-carlo sim EKF: %.2f\n", avg_rmse);
fprintf("Average std RMSE after monte-carlo sim EKF: %.2f\n", std_rmse); 







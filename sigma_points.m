function [X, Wm, Wc] = sigma_points(x, P, alpha, beta, kappa)
    n = numel(x);
    lambda = alpha^2 * (n + kappa) - n;
    Wm = [lambda / (n + lambda); repmat(1 / (2 * (n + lambda)), 2*n, 1)];
    Wc = Wm;
    Wc(1) = Wc(1) + (1 - alpha^2 + beta);
    S = chol((n + lambda) * P, 'lower');
    X = [x, repmat(x, 1, n) + S, repmat(x, 1, n) - S];
end
clc; clear; close all;

% --- Multi-Layer Neural Network for XOR with Detailed Printing ---

% Input patterns and targets for XOR
P = [0 0 1 1; 0 1 0 1];      % Input patterns
T = [-1 1 1 -1];              % Targets (-1 for 0, 1 for 1)

% Network architecture
input_size = 2;
hidden_size = 2;
output_size = 1;
eta = 0.2;                     % Learning rate
max_epochs = 5000;

% Initialize weights and biases randomly
rng('shuffle');
W1 = rand(hidden_size, input_size)*2 - 1;  % Hidden layer weights
b1 = rand(hidden_size,1)*2 - 1;            % Hidden layer biases
W2 = rand(output_size, hidden_size)*2 - 1; % Output layer weights
b2 = rand()*2 - 1;                         % Output layer bias

% Activation function and derivative
f = @(x) tanh(x);        % Hidden and output layer activation
f_prime = @(x) 1 - tanh(x).^2;

% Training loop
for epoch = 1:max_epochs
    total_error = 0;
    fprintf('--- Epoch %d ---\n', epoch);
    
    for i = 1:size(P,2)
        % Forward pass
        x = P(:,i);
        t = T(i);
        
        z1 = W1*x + b1;       % Hidden layer linear combination
        a1 = f(z1);           % Hidden layer output
        z2 = W2*a1 + b2;      % Output layer linear combination
        a2 = f(z2);           % Network output
        
        % Compute error
        e = t - a2;
        total_error = total_error + 0.5*e^2;
        
        % Backpropagation
        delta2 = e .* f_prime(z2);             % Output layer delta
        delta1 = f_prime(z1) .* (W2' * delta2); % Hidden layer delta
        
        % Update weights and biases
        W2 = W2 + eta * delta2 * a1';
        b2 = b2 + eta * delta2;
        W1 = W1 + eta * delta1 * x';
        b1 = b1 + eta * delta1;
        
        % Print detailed calculation
        fprintf('Input %d: [%d %d], Target = %d\n', i, x(1), x(2), t);
        fprintf('  Hidden activations a1 = [%.4f %.4f]\n', a1(1), a1(2));
        fprintf('  Output a2 = %.4f, Error = %.4f\n', a2, e);
        fprintf('  ΔW1 = [%.4f %.4f; %.4f %.4f], Δb1 = [%.4f %.4f]\n', ...
            eta*delta1(1)*x', eta*delta1(2)*x', eta*delta1(1), eta*delta1(2));
        fprintf('  ΔW2 = [%.4f %.4f], Δb2 = %.4f\n\n', eta*delta2*a1', eta*delta2);
    end
    
    fprintf('End of Epoch %d: Total Error = %.6f\n\n', epoch, total_error);
    
    % Stop if error is very small
    if total_error < 1e-5
        fprintf('Training converged at epoch %d\n', epoch);
        break;
    end
end

% Testing
fprintf('\n--- Testing XOR Network ---\n');
for i = 1:size(P,2)
    x = P(:,i);
    z1 = W1*x + b1;
    a1 = f(z1);
    z2 = W2*a1 + b2;
    a2 = f(z2);
    fprintf('Input: [%d %d] -> Output: %.4f -> Target: %d\n', x(1), x(2), a2, T(i));
end

% Plot decision regions
[x_grid, y_grid] = meshgrid(-0.5:0.01:1.5, -0.5:0.01:1.5);
Z = zeros(size(x_grid));
for i = 1:numel(x_grid)
    x = [x_grid(i); y_grid(i)];
    a1 = f(W1*x + b1);
    a2 = f(W2*a1 + b2);
    Z(i) = a2;
end
figure; hold on; grid on;
contourf(x_grid, y_grid, Z, [-1 0 1], 'LineColor','k');
colormap([1 0.8 0.8; 0.8 0.8 1]);
plot(P(1,T==1), P(2,T==1),'ro','MarkerSize',10,'LineWidth',2);
plot(P(1,T==-1), P(2,T==-1),'bx','MarkerSize',10,'LineWidth',2);
xlabel('x_1'); ylabel('x_2'); title('XOR solved with 2-layer Neural Network');
legend('+1','-1');

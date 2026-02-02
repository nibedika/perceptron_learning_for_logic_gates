clc; clear; close all;

% --- Perceptron Learning Rule Demo (Interactive & Animated) ---

% Ask user for truth table
truth_table = input('Enter truth table (AND / OR / XOR): ', 's');
truth_table = upper(truth_table);

switch truth_table
    case 'AND'
        P = [0 0 1 1; 0 1 0 1];      % Input patterns
        T = [-1 -1 -1 1];             % Targets
    case 'OR'
        P = [0 0 1 1; 0 1 0 1];      % Input patterns
        T = [-1 1 1 1];               % Targets
    case 'XOR'
        P = [0 0 1 1; 0 1 0 1];      % Input patterns
        T = [-1 1 1 -1];              % XOR is NOT linearly separable
        fprintf('Note: XOR is not linearly separable. Perceptron will not converge.\n');
    otherwise
        error('Unsupported truth table.');
end

% Initialize weights and bias
rng('shuffle');                  % Random seed
W = rand(1,2)*2 - 1;            % Random weights between -1 and 1
b = rand()*2 - 1;                % Bias
eta = 0.2;                       % Learning rate
max_epochs = 20;

% Plot input points
figure; hold on; grid on;
plot(P(1,T==1), P(2,T==1),'ro','MarkerSize',10,'LineWidth',2); % +1
plot(P(1,T==-1), P(2,T==-1),'bx','MarkerSize',10,'LineWidth',2); % -1
xlabel('x_1'); ylabel('x_2'); axis([-0.5 1.5 -0.5 1.5]);
title(['Perceptron Learning - ', truth_table, ' gate']);

% Perceptron training
for epoch = 1:max_epochs
    errors = 0;
    fprintf('--- Epoch %d ---\n', epoch);
    
    for i = 1:size(P,2)
        xi = P(:,i)';
        ti = T(i);
        
        % Compute output
        oi = sign(W*xi' + b);
        if oi == 0
            oi = -1; % Treat 0 as -1
        end
        
        % Print current calculation
        fprintf('Input %d: xi = [%d %d], t_i = %d, o_i = %d\n', i, xi(1), xi(2), ti, oi);
        
        % Update weights if wrong
        if oi ~= ti
            delta_W = eta*(ti - oi)*xi;
            delta_b = eta*(ti - oi);
            W = W + delta_W;
            b = b + delta_b;
            errors = errors + 1;
            
            fprintf('  ΔW = [%.4f %.4f], Δb = %.4f --> Updated W = [%.4f %.4f], b = %.4f\n', ...
                delta_W(1), delta_W(2), delta_b, W(1), W(2), b);
            
            % Animate decision boundary
            x_plot = -0.5:0.01:1.5;
            y_plot = -(W(1)*x_plot + b)/W(2);
            h = plot(x_plot, y_plot, 'g--', 'LineWidth', 1.5);
            pause(0.5); % Pause to visualize update clearly
            if i < size(P,2) || epoch < max_epochs
                delete(h);  % Remove previous line
            end
        end
    end
    
    fprintf('End of Epoch %d: W = [%.4f %.4f], b = %.4f, errors = %d\n\n', epoch, W(1), W(2), b, errors);
    
    % Stop if no errors (except XOR, which won't converge)
    if errors == 0 && ~strcmp(truth_table,'XOR')
        fprintf('Training converged at epoch %d!\n', epoch);
        break;
    end
end

% Final decision boundary
x_plot = -0.5:0.01:1.5;
y_plot = -(W(1)*x_plot + b)/W(2);
plot(x_plot, y_plot, 'k-', 'LineWidth',2);
legend('+1', '-1', 'Decision Boundary');

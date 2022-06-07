function updates = adadelta(layer, type_update)
%ADADELTA optimization
%   Detailed explanation goes here

if strcmp(type_update, 'w')
    state = layer.w_state;
    gradients = layer.grad.weight;
else
    state = layer.b_state;
    gradients = layer.grad.bias;
end

if isempty(state)
    state = struct;
end

if ~isfield(state, 'epsilon')
    state.epsilon = 1e-6;
end
if ~isfield(state, 'rho')
    state.rho = .95;
end
if ~isfield(state, 'iteration')
    state.iteration = 1;
end
if ~isfield(state, 'history')
    state.history = zeros(size(gradients));
end
if ~isfield(state, 'u')
    state.u = zeros(size(gradients));
end

% accumulate gradient
state.history = state.rho * state.history + (1 - state.rho) * gradients.^2;
    
% update parameters
updates = gradients .* sqrt((state.u + state.epsilon) ./ (state.history + state.epsilon));

% accumulate updates
state.u = state.rho * state.u + (1 - state.rho) * updates.^2;

% update iteration number
state.iteration = state.iteration + 1;

if strcmp(type_update, 'w')
    layer.w_state = state;
else
    layer.b_state = state;
end


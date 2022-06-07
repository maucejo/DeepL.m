function updates = rmsprop(layer, type_update)
%RMSPROP rmsprop optimization
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

if ~isfield(state, 'eta')
    state.eta = 1e-3;
end
if ~isfield(state, 'rho') 
    state.rho = 0.9;
end
if ~isfield(state, 'epsilon')
    state.epsilon = 1e-8;
end
if ~isfield(state, 'iteration')
    state.iteration = 1;
end
if ~isfield(state, 'history')
    state.history = zeros(size(gradients));
end


state.history = state.rho * state.history + (1 - state.rho) * gradients.^2;
    
% update parameters
updates = gradients * state.eta ./ sqrt(state.history + state.epsilon);

% update iteration number
state.iteration = state.iteration + 1;

if strcmp(type_update, 'w')
    layer.w_state = state;
else
    layer.b_state = state;
end


function updates = vanilla(layer, type_update)
%VANILLA The most basic gradient descent
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
    state.eta = 1e-1;
end
if ~isfield(state, 'iteration')
    state.iteration = 1;
end

% compute updates
updates = state.eta * gradients;

% update iteration number
state.iteration = state.iteration + 1;

if strcmp(type_update, 'w')
    layer.w_state = state;
else
    layer.b_state = state;
end


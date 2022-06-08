function updates = amsgrad(layer, type_update)
%AMSGRAD Summary of this function goes here
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

if ~isfield(state, 'beta1')
    state.beta1 = 0.9;
end
if ~isfield(state, 'beta2') 
    state.beta2 = 0.999;
end
if ~isfield(state, 'epsilon')
    state.epsilon = 1e-8;
end
if ~isfield(state, 'iteration')
    state.iteration = 1;
end
if ~isfield(state, 'm')
    state.m = zeros(size(gradients));
end
if ~isfield(state, 'v')
    state.v = zeros(size(gradients));
end
if ~isfield(state, 'vhat')
    state.vhat = zeros(size(gradients));
end
if ~isfield(state, 'eta')
    state.eta = 1e-2;
end

% update biased first moment estimate
state.m = state.beta1 * state.m + (1 - state.beta1) * gradients;
    
% update biased second raw moment estimate
state.v = state.beta2 * state.v + (1 - state.beta2) * gradients.^2;
    
% non-decreasing
state.vhat = max(state.vhat, state.v);
    
% update parameters
updates = state.eta * state.m ./ (sqrt(state.vhat) + state.epsilon);

% update iteration number
state.iteration = state.iteration + 1;

if strcmp(type_update, 'w')
    layer.w_state = state;
else
    layer.b_state = state;
end


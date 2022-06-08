function updates = radam(layer, type_update)
%RADAM Summary of this function goes here
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
if ~isfield(state, 'eta')
    state.eta = 1e-2;
end

rhoinf = 2 / (1 - state.beta2) - 1;

% update biased first moment estimate
state.m = state.beta1 * state.m + (1 - state.beta1) * gradients;
    
% update biased second raw moment estimate
state.v = state.beta2 * state.v + (1 - state.beta2) * gradients.^2;
    
% compute bias-corrected first moment estimate
mhat = state.m / (1 - state.beta1^state.iteration);

% length of the approximated SMA
rho = rhoinf - 2 * state.iteration * state.beta2^state.iteration / (1 - state.beta2^state.iteration);

if rho > 4
    
    % compute bias-corrected second raw moment estimate
    vhat = sqrt(state.v / (1 - state.beta2^state.iteration));
    
    % variance rectification term
    r = sqrt((rho - 4) * (rho - 2) * rhoinf / (rhoinf - 4) / (rhoinf - 2) / rho);
    
    % update parameters
    updates = state.eta * r * mhat ./ (vhat + state.epsilon);
    
else
    
    % update parameters
    updates = state.eta * mhat;
    
end

% update iteration number
state.iteration = state.iteration + 1;

if strcmp(type_update, 'w')
    layer.w_state = state;
else
    layer.b_state = state;
end


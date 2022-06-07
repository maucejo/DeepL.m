classdef Layernorm < matlab.mixin.SetGet
    
    properties (Access = 'public')
        type = 'Layer normalization';
        Nparam
        size_in
        size_out
        gamma
        beta
        momentum = 0.9
        epsilon = 1e-8;
        X_norm
        layer_std
        grad
        w_state
        b_state
    end
    
    methods (Access = 'public')
        % Constructeur
        function self = Batchnorm(size_in)
            self.Nparam = 2*size_in;
            self.gamma = ones(size_in, 1);
            self.beta = zeros(size_in, 1);
            self.size_in = size_in;
            self.size_out = size_in;
        end
        
        function self = init(self)
           self.gamma = [];
           self.beta = [];
           self.grad = [];
           self.layer_std = [];
           self.X_norm = [];
           self.w_state = [];
           self.b_state = [];
        end
        
        % Forward pass
        function out = forward(self, X)
            layer_mean = mean(X, 1);
            layer_var = var(X, 1);
            
            self.layer_std = sqrt(layer_var + self.epsilon);
            self.X_norm = (X - layer_mean)./self.layer_std;
            
            out = self.gamma.*self.X_norm + self.beta;
        end
        
        % Backward pass
        function dX = backward(self, dout)
            
            Nfeatures = size(dout, 1);
            
            dx_norm = self.gamma.*dout;
            dX = (Nfeatures*dx_norm - sum(dx_norm, 1) - sum(dx_norm.*self.X_norm, 1).*self.X_norm)/Nfeatures./self.layer_std;
            
            dgamma = sum(self.X_norm.*dout, 2);
            dbeta = sum(dout, 2);
            
            self.grad.weight = dgamma;
            self.grad.bias = dbeta;
        end
    end
end
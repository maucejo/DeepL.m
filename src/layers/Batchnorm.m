classdef Batchnorm < matlab.mixin.SetGet

    properties (Access = 'public')
        type = 'Batch normalization';
        Nparam
        size_in
        size_out
        gamma
        beta
        momentum = 0.9
        epsilon = 1e-8;
        running_mean
        running_var
        X_norm
        batch_std
        traintest
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
           self.batch_std = [];
           self.running_mean = [];
           self.running_var = [];
           self.X_norm = [];
           self.w_state = [];
           self.b_state = [];
        end

        % Forward pass
        function out = forward(self, X, varargin)

            if isempty(varargin)
                mode = 'train';
            else
                mode = varargin{1};
            end

            if strcmp(mode, 'train')
                batch_mean = mean(X, 2);
                batch_var = var(X, 2);

                if isempty(self.running_mean) && isempty(self.running_var)
                    self.running_mean = batch_mean;
                    self.running_var = batch_var;
                end

                self.running_mean = self.momentum*self.running_mean + (1 - self.momentum)*batch_mean;
                self.running_var = self.momentum*self.running_var + (1 - self.momentum)*batch_var;

                self.batch_std = sqrt(batch_var + self.epsilon);
                self.X_norm = (X - batch_mean)./self.batch_std;

                out = self.gamma.*self.X_norm + self.beta;
            else % test
                b_std = sqrt(self.running_var + self.epsilon);
                Xnorm = (X - self.running_mean)./b_std;

                out = self.gamma.*Xnorm + self.beta;
            end
        end

        % Backward pass
        function dX = backward(self, dout)

            Nsamples = size(dout, 2);

            dx_norm = self.gamma.*dout;
            dX = (Nsamples*dx_norm - sum(dx_norm, 2) - sum(dx_norm.*self.X_norm, 2).*self.X_norm)/Nsamples./self.batch_std;

            dgamma = sum(self.X_norm.*dout, 2);
            dbeta = sum(dout, 2);

            self.grad.weight = dgamma;
            self.grad.bias = dbeta;
        end
    end
end
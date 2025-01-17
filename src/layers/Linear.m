classdef Linear < matlab.mixin.SetGet

    properties (Access = 'public')
        type = 'Activation - Linear';
        Nparam
        grad
    end

    methods (Access = 'public')
        % Constructeur
        function self = Linear()
            self.Nparam = 0;
        end

        function self = init(self)
           return;
        end

        % Forward pass
        function out = forward(self, X)
            out = X;
        end

        % Backward pass
        function dX = backward(self, dout)
            dX = dout;
            self.grad = {};
        end
    end
end
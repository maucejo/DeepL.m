classdef Tanh < matlab.mixin.SetGet
   
    properties (Access = 'public')
        type = 'Activation - Tanh';
        Nparam
        out
        grad
    end
    
    methods (Access = 'public')
        % Constructeur
        function self = Tanh()
            self.Nparam = 0; 
        end
        
        function self = init(self)
           self.out = []; 
        end
        
        % Forward pass
        function out = forward(self, X)
            out = tanh(X);
            self.out = out;
        end
        
        % Backward pass
        function dX = backward(self, dout)
            dX = (1 - self.out.^2).*dout;
        end
    end
end
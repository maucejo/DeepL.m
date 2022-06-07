classdef Sigmoid < matlab.mixin.SetGet
   
    properties (Access = 'public')
        type = 'Activation - sigmoid';
        Nparam
        out
        grad
    end
    
    methods (Access = 'public')
        % Constructeur
        function self = Sigmoid()
            self.Nparam = 0; 
        end
        
        function self = init(self)
           self.out= []; 
        end
        
        % Forward pass
        function out = forward(self, X)
            out = 1./(1 + exp(X));
            self.out = out;
        end
        
        % Backward pass
        function dX = backward(self, dout)
            dX = self.out.*(1 - self.out).*dout;
        end
    end
end
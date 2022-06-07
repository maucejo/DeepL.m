classdef Relu < matlab.mixin.SetGet
   
    properties (Access = 'public')
        type = 'Activation - Relu';
        Nparam
        X
        grad
    end
    
    methods (Access = 'public')
        % Constructeur
        function self = Relu()
            self.Nparam = 0; 
        end
        
        function self = init(self)
           self.X = []; 
        end
        
        % Forward pass
        function out = forward(self, X)
            out = max(0, X);
            self.X = X;
        end
        
        % Backward pass
        function dX = backward(self, dout)
            dX = dout;
            dX(self.X <= 0) = 0;
        end
    end
end
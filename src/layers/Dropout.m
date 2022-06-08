classdef Dropout < matlab.mixin.SetGet
   
    properties (Access = 'public')
       type = 'Dropout'
       drop_rate
       mask
       Nparam
       traintest
       grad
    end
    
    methods (Access = 'public')
       % Constructeur
       function self = Dropout(type, drop_rate)
           self.type = type;
           self.drop_rate = drop_rate;
           self.Nparam = 0;
       end
       
       function self = init(self)
           self.mask = [];
       end
       
       % Forward pass
       function out = forward(self, X, varargin)
           Nx = size(X, 1);
           
           if isempty(varargin)
               mode = 'train';
           else
               mode = varargin{1};
           end
           
           if strcmp(mode, 'train')
               if strcmp(self.type, 'classic')
                   self.mask = (rand(Nx, 1) <= (1 - self.drop_rate))/(1 - self.drop_rate);
               else
                   self.mask = 1 + sqrt(self.drop_rate/(1 - self.drop_rate))*randn(Nx, 1);
               end
               
               out = self.mask.*X;
           else
               out = X;
           end
       end
       
       % Backward pass
       function dX = backward(self, dout)
           dX = dout.*self.mask;
       end
    end
end
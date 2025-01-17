classdef Dense < matlab.mixin.SetGet

    properties (Access = 'public')
        type = 'Dense';
        Nparam
        size_in
        size_out
        W % Matrice des poids de la couche
        b % Vecteur des biais de la couche
        X % Vecteur d'entrée de la couche
        grad
        w_state
        b_state
    end

    methods (Access = 'public')
        % Constructeur
        function self = Dense(size_in, size_out)
            % Glorot initialization
            r = sqrt(6/(size_in + size_out));
            self.W = -r + 2*r*rand(size_in, size_out); % Loi uniforme
            self.b = zeros(size_out, 1);
            self.Nparam = numel(self.W) + numel(self.b);
            self.size_in = size_in;
            self.size_out = size_out;
        end

        function self = init(self)
            r = sqrt(6/(self.size_in + self.size_out));
            self.W = -r + 2*r*rand(self.size_in, self.size_out); % Loi uniforme
            self.b = zeros(self.size_out, 1);
            self.grad = [];
            self.w_state = [];
            self.b_state = [];
            self.X = [];
        end

        % Forward pass
        function out = forward(self, X)
           self.X = X;
           out = self.W.'*self.X + self.b;
        end

        % Backward pass
        function dX = backward(self, dout)
            dW = self.X*dout.';
            db = sum(dout, 2);
            dX = self.W*dout;

            self.grad.weight = dW;
            self.grad.bias = db;
        end
    end
end
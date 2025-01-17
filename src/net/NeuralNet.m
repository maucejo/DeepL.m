classdef NeuralNet < matlab.mixin.SetGet

    properties (Access = 'public')
        layers             % Cellule contenant les couches

        eta = 1e-3;        % Learning rate
        minLoss = 1e-6;    % Tolérance pour la convergence
        max_epoch = 5e4;   % Nombre d'époque max
        reg = false        % Activation de la régularisation
        shuffle = true;    % Mélange des données
        reg_type           % Type de régularisation
        lambda             % Paramètre de régularisation
        batch_size         % Taille du batch
        mini_batch         % Définition des mini-batches
        istrained = false  % Indicateur d'entraînement
        Nlayers
        optimizer
        epoch
        monitor = false;
        loss
        loss_train
        loss_type = 'sse';
        metrics_train
        metrics_test
        metrics_type = 'mse';
        grad_update
    end
    methods (Access = 'public')
        %% Constructeur
        function self = NeuralNet(varargin)

            if ~isempty(varargin)
                self.layers = varargin{1};
                self.Nlayers = length(self.layers);
            else
                self.Nlayers = 0;
            end
        end

        function self = init(self)
            for ee = 1:self.Nlayers
               self.layers{ee}.init;
            end
        end

        %% Ajout d'une couche
        function self = add(self, layer, varargin)
            self.Nlayers = self.Nlayers + 1;

            if isempty(varargin)
                self.layers{self.Nlayers} = layer;
            else
                ID_layer = varargin{1};

                if ID_layer > (self.Nlayers - 1)
                    self.layers{self.Nlayers} = layer;
                else
                   new_layers = cell(1, self.Nlayers);

                   for ee = 1:self.Nlayers
                       if ee < ID_layer
                           new_layers{ee} = self.layers{ee};
                       elseif ee == ID_layer
                           new_layers{ee} = layer;
                       else
                           new_layers{ee} = self.layers{ee - 1};
                       end
                   end

                   self.layers = new_layers;
                end
            end
        end

        %% Enlèvement d'une couche
        function self = remove(self, varargin)
            if isempty(varargin)
                ID_rem_layer = self.Nlayers;
            else
                ID_rem_layer = varargin{1};
            end

            if isempty(self.layers)
                disp('Le réseau est vide')
                return;
            end

            if ID_rem_layer > self.Nlayers
                ID_rem_layer = self.Nlayers;
                disp('La dernière couche du réseau a été enlevée')
            end

            self.layers(ID_rem_layer) = [];
            self.Nlayers = length(self.layers);
        end

        %% Vérification de la cohérence du modèle
        function self = consistency(self)
            for ee = 1:self.Nlayers
               type = self.layers{ee}.type(1:3);
               if strcmp(type, 'Act') || strcmp(type, 'Dro')
                   continue;
               else
                   if ee == 1
                      size_out_l_1 = self.layers{ee}.size_out;
                      Idl_out_l_1 = ee;
                   else
                      size_in_l = self.layers{ee}.size_in;
                      Idl_in_l = ee;
                      if size_out_l_1 == size_in_l
                         size_out_l_1 = self.layers{ee}.size_out;
                         Idl_out_l_1 = Idl_in_l;
                      else
                          error(['Le réseau n''est pas consistent au niveau des couches ' num2str(Idl_out_l_1) ' et ' num2str(Idl_in_l) newline ...
                              'Nombre de neurones de la couche ' num2str(Idl_out_l_1) ' : ' num2str(size_out_l_1) newline ...
                              'Nombre de neurones de la couche ' num2str(Idl_in_l) ' : ' num2str(size_in_l)]);
                      end
                   end
               end
            end
            disp('Les dimensions du réseau sont consistentes')
        end

        %% Résumé du modèle
        function summary(self)
            varTypes = ["int64", "string", "int64"];
            varNames = ["Layer ID", "Type", "Trainable parameters"];
            T = table('Size', [self.Nlayers, 3], 'VariableTypes', varTypes, 'VariableNames', varNames);

            Nparam_tot = 0;
            for ee = 1:self.Nlayers
                Nparam_layer = self.layers{ee}.Nparam;
                Nparam_tot = Nparam_tot + Nparam_layer;

                T(ee, :) = {ee, self.layers{ee}.type, Nparam_layer};
            end
            disp(T)
            disp(['Number of trainable parameters: ' num2str(Nparam_tot)])
        end

        %% Entraînement
        function self = train(self, Xtrain, Ytrain, options, varargin)

            % Initialisation
            Nsamples = size(Xtrain, 2);

            if nargin == 3
                self.batch_size = Nsamples;
                self.optimizer = 'adabelief';
                self.grad_update = @adabelief;
                loss_func = @sse_loss;
                metrics_func = @mse_metrics;
            elseif nargin > 3
                if isfield(options, 'eta')
                    self.eta = options.eta;
                end

                if isfield(options, 'minLoss')
                    self.minLoss = options.minLoss;
                end

                if isfield(options, 'max_epoch')
                    self.max_epoch = options.max_expoch;
                end

                if isfield(options, 'batch_size')
                    self.batch_size = options.batch_size;
                end

                if isfield(options, 'reg')
                    self.reg = options.reg;

                    if self.reg
                        if isfield(options.reg_opt, 'type')
                            self.reg_type = lower(options.reg_opts.type);
                        else
                            self.reg_type = 'l2';
                        end

                        reg_func_name = [self.reg_type '_reg_loss'];
                        reg_loss_func = str2func(reg_func_name);

                        reg_backward_name = [self.reg_type '_backward'];
                        self.backward_reg = str2func(reg_backward_name);

                        if isfield(options.reg_opt, 'lambda')
                            self.lambda = options.reg_opt.lambda;
                        else
                            if strcmp(self.reg_type, 'l2') || strcmp(self.reg_type, 'l1')
                                self.lambda = 1e-3;
                            else % l1-l2
                                self.lambda = 1e-3*ones(1, 2);
                            end
                        end
                    end
                end

                if isfield(options, 'batch_size')
                    self.batch_size = options.batch_size;
                else
                    self.batch_size = Nsamples;
                end

                if isfield(options, 'shuffle')
                    self.shuffle = options.shuffle;
                end

                if isfield(options, 'optimizer')
                    self.optimizer = lower(options.optimizer);
                else
                    self.optimizer = 'adabelief';
                end
                self.grad_update = str2func(self.optimizer);

                if isfield(options, 'loss_type')
                    self.loss_type = lower(options.loss_type);
                end
                loss_name = [self.loss_type, '_loss'];
                loss_func = str2func(loss_name);

                if isfield(options, 'metrics_type')
                    self.metrics_type = lower(options.metrics_type);
                end
                metrics_name = [self.metrics_type, 'metrics'];
                metrics_func = str2func(metrics_name);

                if ~isempty(varargin)
                    if nargin ~= 2
                       error('Il faut donner Xtest ET Ytest')
                    else
                        Xtest = varargin{1};
                        Ytest = varargin{2};
                    end
                end

                if isfield(options, 'monitor')
                    self.monitor = options.monitor;
                end
            end

            if ~self.shuffle
                Xs = Xtrain;
                Ys = Ytrain;
            end

            if self.monitor
                self.loss_train = nan(self.max_epoch, 1);
                self.metrics_train = nan(self.max_epoch, 1);
                if ~isempty(varargin)
                    self.metrics_test = nan(self.max_epoch, 1);
                end
            end

            % Création des mini-batches
            [idx_mb, Nmini_batch] = self.create_mini_batch(Nsamples);

            % Apprentissage
            self.epoch = 1;
            self.loss = Inf;
            while (self.epoch <= self.max_epoch) && (self.loss >= self.minLoss)
                for ee = 1:Nmini_batch
                    if self.shuffle
                        idx = randperm(Nsamples, Nsamples);
                        Xs = Xtrain(:, idx);
                        Ys = Ytrain(:, idx);
                    end

                    Xs = Xs(:, idx_mb{ee});
                    Ys = Ys(:, idx_mb{ee});

                    % 1. Forward pass
                    out = self.pred(Xs, 'train');

                    % 2. Calcul de la fonctionnelle d'erreur
                    [self.loss, dout] = loss_func(out, Ys);

                    if self.reg
                       reg_loss = reg_loss_func(self.layers, self.lambda);
                       self.loss = self.loss + reg_loss;
                    end

                    % 3. Backward pass
                    self.backward(dout);

                    % 4. Mise à jour des gradients
                    self.update
                end

                % 5. Monitoring
                 self.loss_train(self.epoch) = self.loss;
                 self.metrics_train(self.epoch) = metrics_func(out, Ys);

                if ~isempty(varargin)
                    out_test = self.pred(Xtest);
                    self.metric_test(self.epoch) = metrics_func(out_test, Ytest);
                end

                if self.monitor
                    if rem(self.epoch, 50) == 0
                        semilogy(1:self.epoch, self.loss_train(1:self.epoch), 'b')
                        hold on
                        semilogy([1 self.epoch], [self.minLoss self.minLoss], 'r')
                        hold off
                        axis tight
                        pause(0.01)
                    end
                end

                % 6. Fin d'une époque
                self.epoch = self.epoch + 1;
            end

            % Fin apprentissage
            self.epoch = self.epoch - 1;

            self.loss_train = self.loss_train(1:self.epoch);
            self.metrics_train = self.metrics_train(1:self.epoch);
            if ~isempty(varargin)
                self.metrics_test = self.metrics_test(1:self.epoch);
            end

            self.istrained = true;
        end

        %% Prédiction du modèle
        function X = pred(self, X, varargin)
            if isempty(varargin)
                mode = 'test';
            else
                mode = varargin{1};
            end

            for ee = 1:self.Nlayers
                if isprop(self.layers{ee}, 'traintest')
                    X = self.layers{ee}.forward(X, mode);
                else
                    X = self.layers{ee}.forward(X);
                end
            end
        end

        %% Rétro-propagation des gradients
         function backward(self, dout)
             for ee = self.Nlayers:-1:1
                 dout = self.layers{ee}.backward(dout);
                 if self.reg
                     backward_reg(self.layers{ee}, self.lambda);
                 end

                 if self.layers{ee}.Nparam > 0
                    self.layers{ee}.w_state.eta = self.eta;
                    self.layers{ee}.w_state.max_iter = self.max_epoch;

                    self.layers{ee}.b_state.eta = self.eta;
                    self.layers{ee}.b_state.max_iter = self.max_epoch;
                 end
             end
         end

         %% Mise à jour du réseau
         function update(self)
             for ee = self.Nlayers:-1:1
                 if ~isempty(self.layers{ee}.grad)
                    Wu = self.grad_update(self.layers{ee}, 'w');
                    bu = self.grad_update(self.layers{ee}, 'b');

                    layer_type = self.layers{ee}.type(1:5);
                    if strcmp(layer_type, 'Dense')
                        self.layers{ee}.W = self.layers{ee}.W - Wu;
                        self.layers{ee}.b = self.layers{ee}.b - bu;
                    elseif strcmp(layer_type, 'Batch') || strcmp(layer_type, 'Layer')
                        self.layers{ee}.gamma = self.layers{ee}.gamma - Wu;
                        self.layers{ee}.beta = self.layers{ee}.beta - bu;
                    end
                 end
             end
         end

        %% Création des mini-batches
        function [idx_mb, Nmini_batch] = create_mini_batch(self, Nsamples)
            Nmini_batch = ceil(Nsamples/self.batch_size);
            idx_mb = cell(Nmini_batch, 1);

            if Nmini_batch >= 1
                idx_mb{1} = 1:Nsamples;
            else
                Nmb = 0;
                idx = 1:self.batch_size;
                for ee = 1:Nmini_batch
                    if ee < Nmini_batch
                        self.idx_mb{ee} = Nmb + idx;
                        Nmb = Nmb + self.batch_size;
                    else
                        Nmb_last = Nsamples - Nmb;
                        idx_mb{ee} = Nmb + (1:Nmb_last);
                    end
                end
            end
        end
    end
end

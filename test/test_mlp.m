%% Création d'un réseau de neurones - Version 1
layers = {
    Dense(1, 8)
    Linear()
    Dense(8, 4)
    Tanh()
    Dense(4, 1)
    Linear()
};
ann = NeuralNet(layers);

%% Création d'un réseau de neurones - Version 2
ann2 = NeuralNet();

ann2.add(Dense(1, 8));
ann2.add(Linear());
ann2.add(Dense(8, 4));
ann2.add(Tanh());
ann2.add(Dense(4, 1));
ann2.add(Linear());

%% Comparaison des deux méthodes de construction
ann.summary
ann2.summary

%% Vérification de la consistence des réseaux
ann.consistency;
ann2.consistency;

%% Initialisation des réseaux - On utilise que le réseau 1
ann.init;

%% Exemple approximation d'une fonction
% func = 'abs';
func = 'heaviside';
% func = 'const';
% func = 'sin';
% func = 'triangle';
% func = 'x2';

n = 25;
X = linspace(-5, 5, n);
Xpred = linspace(-5, 5, 2*n);

switch func
    case 'abs'
        Y = abs(X);
        Ytrue = abs(Xpred);
    case 'heaviside'
        Y = zeros(1, n);
        Ytrue = zeros(1, 2*n);
        Y(X <= 0) = -1;
        Y(X > 0) = 1;
        Ytrue(Xpred <= 0) = -1;
        Ytrue(Xpred > 0) = 1;
    case 'const'
        Y = ones(1, n);
        Ytrue = ones(1, 2*n);
    case 'sin'
        Y = sin(X);
        Ytrue = sin(Xpred);
    case 'triangle'
        Y = zeros(1, n);
        Ytrue = zeros(1, 2*n);
        Y(X < 0 & X >= -1) = 1 + X(X < 0 & X >= -1);
        Y(X >= 0 & X <= 1) = 1 - X(X >= 0 & X <= 1);
        Ytrue(Xpred < 0 & Xpred >= -1) = 1 + Xpred(Xpred < 0 & Xpred >= -1);
        Ytrue(Xpred >= 0 & Xpred <= 1) = 1 - Xpred(Xpred >= 0 & Xpred <= 1);
    case 'x2'
        Y = X.^2;
        Ytrue = Xpred.^2;
end

Xtrain = X;
Ytrain = Y;

%% Entraînement du réseau
ann.max_epoch = 1e4;
ann.train(Xtrain, Ytrain);
% ann2.train(Xtrain, Ytrain);

%% Prédiction
Ypred = ann.pred(Xpred);

%% Affichage
figure;
hold on
plot(Xpred, Ytrue, 'b', 'linewidth', 2, 'DisplayName', 'Fonction à approximer');
plot(Xpred, Ypred, '--r', 'linewidth', 2, 'DisplayName', 'Prédiction');
plot(Xtrain, Ytrain, 'ok', 'MarkerSize', 10, 'DisplayName', 'Points d''entraînement');
legend()
hold off
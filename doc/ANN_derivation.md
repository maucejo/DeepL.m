# Réseau de neurones artificiels pour des problèmes de régression

Cette documentation vise à présenter les différentes équations régissant les réseaux de neurones artificiels adaptés à la résolution de problèmes de régression

## 1. Perceptron simple couche

Considérons le perceptron simple couche présenté en Figure 1.

<img src=".\perceptron.png" alt="img" style="zoom: 80%;" />

<center><b>Figure 1 :</b> Perceptron simple couche (source: D. Stansbury - The Clever Machine)</center>

Pour un ensemble d'entrée $a_i$ (pour $i = 1, \dots, N$), le sortie $a_{out}$ d'un perceptron simple couche est donnée par l'équation suivante:
$$
a_{out} = g(z) = g\left(b + \sum_{i = 1}^N a_i w_i\right)
$$
où:

* $w_i$ est le poids associé à l'entrée $a_i$
* $b$ est un biais
* $z = b + \sum_{i = 1}^N a_i w_i$ est la valeur de pré-activation du neurone de sortie
* $g(z)$ est la fonction d'activation du neurone de sortie   

## 2. Perceptron multicouche - Généralités

Considérons le perceptron à une couche cachée (i.e. un perceptron à deux couches) présenté en Figure 2.

<img src=".\multi-layer-perceptron.png" style="zoom: 50%;" />

<center><b>Figure 2 :</b> Perceptron à deux couches (source: D. Stansbury - The Clever Machine)</center>

​	Ici la valeur de sortie du perceptron s'écrit :
$$
\begin{split}
a_k^{(2)} &= g_k^{(2)}\left(z_k^{(2)}\right) \\
&= g_k^{(2)}\left(b_k^{(2)} + \sum_j a_j^{(1)}w_{jk}^{(1,2)}\right) \\
&= g_k^{(2)}\left(b_k^{(2)} + \sum_j g_j^{(1)}\left(z_j^{(1)}\right)w_{jk}^{(1,2)}\right) \\
&= g_k^{(2)}\left(b_k^{(2)} + \sum_j g_j^{(1)}\left(b_j^{(1)} + \sum_ia_i^{(0)}w_{ij}^{(0,1)}\right)w_{jk}^{(1,2)}\right)
\end{split}
$$
avec :

* $z_j^{(l)}$ : Fonction de pré-activation du neurone $j$ de la couche $l$ 
* $g_j^{(l)}$ : Fonction d'action du neurone $j$ de la couche $l$
* $a_j^{(l)}$ : Sortie du neurone $j$ de la couche $l$
* $w_{ij}^{(l-1,l)}$ : Poids entre le neurone $i$ de la couche $l-1$ et le neurone $j$ de la couche $l$
* $b_j^{(l)}$ : Biais du neurone $j$ de la couche $l$ 

Pour les problèmes de régression, la fonction d'activation du neurone de sortie est généralement la fonction linéaire : 
$$
g_k(x) = x \; \textsf{ et }\;  g_k'(x) = 1
$$
tandis que pour les couches cachées, on choisi généralement la fonction tangente hyperbolique :
$$
g_j(x) = \tanh(x) \; \textsf{ et } \; g_j'(x) = 1 - g_j(x)^2
$$

Pour définir complètement le perceptron multicouche, il faut donc calculer les poids $w_{ij}^{(l-1,l)}$ et les biais $b_j^{(l)}$ étant donné un ensemble d'apprentissage $\mathcal{S}_{train} = (\mathcal{X}_{train}, \mathcal{Y}_{train})$. Dans la définition de l'ensemble d'apprentissage, $\mathcal{X}_{train}$ correspond à la couche d'entrée $a_i^{(0)}$ (pour $i = 1, \dots, N_0$), tandis que $\mathcal{Y}_{train}$ correspond à la couche de sortie $a_k^{L}$ (pour $k = 1, \dots, N_L$). Si $N_e$ est le nombre d'exemple utilisé pour réaliser l'apprentissage, alors $\mathcal{X}_{train}$ et $\mathcal{Y}_{train}$ ont alors respectivement pour dimension $N_0 \times N_e$ et $N_L \times N_e$.

## 3. Perceptron multicouche - Apprentissage

Pour apprendre les poids $w_{ij}^{(l-1,l)}$ et les biais $b_j^{(l)}$ à partir de l'ensemble d'apprentissage $\mathcal{S}_{train}$, on utilise généralement l'algorithme de rétropropagation du gradient, basé sur une approche par descente de gradient.

Plus précisément, la méthode repose sur les 4 étapes suivantes :

1. Propagation des données contenues dans $\mathcal{X}_{train}$ de la couche d'entrée 0 vers la couche de sortie $L$ étant donné des valeurs initiales des poids $w_{ij}^{(l-1,l)}$ et des biais $b_j^{(l)}$. Le choix de ces valeurs initiales nécessite la mise en place d'une procédure d'initialisation adaptée. En pratique, celle-ci dépend de la fonction d'activation utilisée.
2. Calcul des paramètres de rétropropagation des gradients.
3. Mise à jour des poids $w_{ij}^{(l-1,l)}$ et des biais $b_j^{(l)}$ par descente de gradient.
4. Répéter les étapes 1 à 3 pour un nombre $N_{ep}$ d'époques correspondant à un cycle complet propagation/rétropropagation pour l'ensemble $\mathcal{S}_{train}$ en prenant comme valeurs initiales les poids et les biais mis à jour.

### 3.1. Initialisation

Comme indiqué en préambule, le choix des valeurs initiales des poids $w_{ij}^{(l-1,l)}$ et des biais $b_j^{(l)}$ dépend de celui de la fonction d'activation. Dans le cas d'un problème de régression pour lequel on utilise généralement la fonction d'activation $\tanh$, on utilise la procédure d'initialisation dite de Xavier (d'après Xavier Glorot, ingénieur chez DeepMind). Formellement, cette initialisation s'écrit :
$$
w_{ij}^{(l-1,l)} \sim \mathcal{N}\left(0, \frac{2}{N_{l-1} + N_l}\right) \; \textsf{ et } \; b_j^{(l)} = 0
$$
Ce type d'initialisation permet de casser les symétries au sein du réseau et facilite ainsi la phase d'apprentissage (en terme de convergence notamment).

### 3.2. Calcul des paramètres de rétropropagation des gradients

Les poids $w_{ij}^{(l-1,l)}$ et les biais $b_j^{(l)}$ sont choisis de manière à minimiser une certaine fonctionnelle. Dans le cas d'un problème de régression, on cherche à minimiser l'erreur quadratique totale entre les données d'entraînement contenues dans $\mathcal{Y}_{train}$, notées $y_{k,n}$ (pour $n = 1, \dots, N_e$), et les valeurs de sortie du réseau, à savoir $a_{k,n}^{(L)}$. Mathématiquement, cette fonctionnelle s'écrit donc :
$$
E(\mathbf{w}, \mathbf{b}) = \frac{1}{2}\sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)^2
$$
**Calcul des gradients de la fonctionnelle d'erreur pour la couche de sortie**

Le calcul du gradient de la fonctionnelle d'erreur par rapport aux poids de la dernière couche s'écrit :
$$
\begin{split}
\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial w_{jk}^{(L-1, L)}} &= \frac{1}{2}\sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\frac{\partial\bigl(a_{k,n}^{(L)} - y_{k,n}\bigr)}{\partial w_{jk}^{(L-1, L)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)\frac{\partial a_{k,n}^{(L)}}{\partial w_{jk}^{(L-1, L)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)\frac{\partial g_k^{(L)}\bigl(z_{k,n}^{(L)}\bigr)}{\partial w_{jk}^{(L-1, L)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)\frac{\partial z_{k,n}^{(L)}}{\partial w_{jk}^{(L-1, L)}} \\
&= \sum_{n = 1}^{N_e}\left(a_{k,n}^{(L)} - y_{k,n}\right){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)a_{j,n}^{(L-1)}
\end{split}
$$
Pour aller plus loin, on note $\delta_{k,n}^{(L)} = \bigl(a_{k,n}^{(L)} - y_{k,n}\bigr){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)$. Ce faisant, le gradient de la fonctionnelle d'erreur par rapport aux poids de la dernière couche devient :
$$
\fbox{$\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial w_{jk}^{(L-1, L)}} = \sum_{n = 1}^{N_e}\delta_{k,n}^{(L)}a_{j,n}^{(L-1)}$}
$$


On procède de même pour le calcul des gradients de la fonctionnelles d'erreur par rapport aux biais.
$$
\begin{split}
\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial b_k^{(L)}} &= \frac{1}{2}\sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\frac{\partial\bigl(a_{k,n}^{(L)} - y_{k,n}\bigr)}{\partial b_k^{(L)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)\frac{\partial a_{k,n}^{(L)}}{\partial b_k^{(L)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)\frac{\partial g_k^{(L)}\bigl(z_{k,n}^{(L)}\bigr)}{\partial b_k^{(L)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)\frac{\partial z_{k,n}^{(L)}}{\partial b_k^{(L)}} \\
&= \sum_{n = 1}^{N_e}\left(a_{k,n}^{(L)} - y_{k,n}\right){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)
\end{split}
$$
En se rappelant que $\delta_{k,n}^{(L)} = \bigl(a_{k,n}^{(L)} - y_{k,n}\bigr){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)$, on trouve finalement :
$$
\fbox{$\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial b_k^{(L)}} = \sum_{n = 1}^{N_e} \delta_{k,n}^{(L)}$}
$$
**Calcul des gradients de la fonctionnelle d'erreur pour les couches cachées**

Le calcul du gradient de la fonctionnelle d'erreur par rapport aux poids de la couche cachée $L-1$ s'écrit :
$$
\begin{split}
\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial w_{ij}^{(L-2, L-1)}} &= \frac{1}{2}\sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\frac{\partial\bigl(a_{k,n}^{(L)} - y_{k,n}\bigr)}{\partial w_{ij}^{(L-2, L-1)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)\frac{\partial a_{k,n}^{(L)}}{\partial w_{ij}^{(L-1, L)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)\frac{\partial g_k^{(L)}\bigl(z_{k,n}^{(L)}\bigr)}{\partial w_{ij}^{(L-2, L-1)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)\frac{\partial g_k^{(L)}\bigl(z_{k,n}^{(L)}\bigr)}{\partial w_{ij}^{(L-2, L-1)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)\frac{\partial z_{k,n}^{(L)}}{\partial w_{ij}^{(L-2, L-1)}} \\
\end{split}
$$
or : 
$$
\begin{split}
\frac{\partial z_{k,n}^{(L)}}{\partial w_{ij}^{(L-2, L-1)}} &= \frac{\partial z_{k,n}^{(L)}}{\partial a_{j,n}^{(L-1)}}\frac{\partial a_j^{(L-1)}}{\partial w_{ij}^{(L-2,L-1)}} \\
&= \frac{\partial \bigl(b_k^{(L)} + \sum_j a_{j,n}^{(L-1)}w_{jk}^{(L-1,L)}\bigr)}{\partial a_{j,n}^{(L-1)}}\frac{\partial g_j^{(L-1)}\bigl(z_j^{(L-1)}\bigr)}{\partial w_{ij}^{(L-2, L-1)}} \\
&= w_{jk}^{(L-1,L)}{g_j^{(L-1)}}'\bigl(z_j^{(L-1)}\bigr)a_{i,n}^{(L-2)}
\end{split}
$$
Par conséquent :
$$
\begin{split}
\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial w_{ij}^{(L-2, L-1)}} &= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)w_{jk}^{(L-1,L)}{g_j^{(L-1)}}'\bigl(z_j^{(L-1)}\bigr)a_{i,n}^{(L-2)} \\
&= \sum_{n = 1}^{N_e}\left[\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)w_{jk}^{(L-1,L)}\right]{g_j^{(L-1)}}'\bigl(z_j^{(L-1)}\bigr)a_{i,n}^{(L-2)} \\
&= \sum_{n = 1}^{N_e}\left[\sum_{k = 1}^{N_k}\delta_{k,n}^{(L)} w_{jk}^{(L-1,L)}\right]{g_j^{(L-1)}}'\bigl(z_j^{(L-1)}\bigr)a_{i,n}^{(L-2)}
\end{split}
$$
En posant alors $\delta_j^{(L-1)} = \left[\sum_{k = 1}^{N_k}\delta_{k,n}^{(L)} w_{jk}^{(L-1,L)}\right]{g_j^{(L-1)}}'\bigl(z_j^{(L-1)}\bigr)$, on trouve finalement :
$$
\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial w_{ij}^{(L-2, L-1)}} = \sum_{n = 1}^{N_e}\delta_{j,n}^{(L-1)}a_{i,n}^{(L-2)}
$$
On en déduit donc pour une couche caché $l$ quelconque :
$$
\fbox{$\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial w_{ij}^{(l-1, l)}} = \sum_{n = 1}^{N_e}\delta_{j,n}^{(l)}a_{i,n}^{(l-1)}$}
$$
On procède de même pour le calcul des gradients de la fonctionnelles d'erreur par rapport aux biais.
$$
\begin{split}
\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial b_j^{(L-1)}} &= \frac{1}{2}\sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\frac{\partial\bigl(a_{k,n}^{(L)} - y_{k,n}\bigr)}{\partial b_j^{(L-1)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)\frac{\partial a_{k,n}^{(L)}}{\partial b_j^{(L-1)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right)\frac{\partial g_k^{(L)}\bigl(z_{k,n}^{(L)}\bigr)}{\partial b_j^{(L-1)}} \\
&= \sum_{n = 1}^{N_e}\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)\frac{\partial z_{k,n}^{(L)}}{\partial b_j^{(L-1)}} 
\end{split}
$$
or :
$$
\begin{split}
\frac{\partial z_{k,n}^{(L)}}{\partial b_j^{(L-1)}} &= \frac{\partial z_{k,n}^{(L)}}{\partial a_{j,n}^{(L-1)}}\frac{\partial a_{j,n}^{(L-1)}}{\partial b_j^{(L-1)}} \\
&= w_{jk}^{(L-1,L)}{g_j^{(L-1)}}'\bigl(z_{j,n}^{(L-1)}\bigr)
\end{split}
$$
Par conséquent :
$$
\begin{split}
\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial b_j^{(L-1)}} &= \sum_{n = 1}^{N_e}\left[\sum_{k = 1}^{N_k}\left(a_{k,n}^{(L)} - y_{k,n}\right){g_k^{(L)}}'\bigl(z_{k,n}^{(L)}\bigr)w_{jk}^{(L-1,L)}\right]{g_j^{(L-1)}}'\bigl(z_{j,n}^{(L-1)}\bigr) \\
&= \sum_{n = 1}^{N_e}\left[\sum_{k = 1}^{N_k}\delta_{k,n}^{(L)} w_{jk}^{(L-1,L)}\right]{g_j^{(L-1)}}'\bigl(z_{j,n}^{(L-1)}\bigr) \\
&= \sum_{n = 1}^{N_e} \delta_{j,n}^{(L-1)}
\end{split}
$$
On en déduit donc pour une couche cachée $l$ quelconque :
$$
\fbox{$\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial b_j^{(l)}} = \sum_{n = 1}^{N_e} \delta_{j,n}^{(l)}$}
$$

### 3.3. Mise à jours des poids et des biais

La mise à jour des poids et des biais s'effectue grâce à une passe de descente de gradient pour un pas $\eta$, c'est-à-dire :
$$
\begin{split}
w_{ij}^{(l-1,l)} &\leftarrow w_{ij}^{(l-1,l)} - \eta\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial w_{ij}^{(l-1, l)}} \\
b_j^{(l)} &\leftarrow b_j^{(l)} - \eta\frac{\partial E(\mathbf{w}, \mathbf{b})}{\partial b_j^{(l)}}
\end{split}
$$

### 3.4. Bilan

<img src=".\backpropagation-steps.png" alt="img" style="zoom:50%;" />
<center><b>Figure 3 :</b> Résumé de la procédure de rétropropagation du gradient </br> (source: D. Stansbury - The Clever Machine)</center>

## 4.  Implémentation - Entraînement

```matlab
function [W, b, E] = Train_MultiLayer_Perceptron(X, Y, hidden_layer, options)
% Entrées:
%   * X: Matrice des données d'entrées: dimensions (N0 x Ne)
%		 N0: Nombre de neurones de la couche d'entrée
%		 Ne: Nombre d'exemple d'entraînement
%	* Y: Matrice des données de sorties: dimensions (NL x Ne)
%		 NL: Nombre de neurones de la couche de sortie
%   * hidden_layer: Vecteur contenant le nombre de neurones pour chaque couche cachée
%        Exemple: hidden_layer = [3, 4] - Deux couches cachées contenant
%			      respectivement 3 et 4 neurones
%	* options: Structure contenant différentes options
%		- eta: Pas de l'algorithme de descente
%		- tol: Tolérance d'arrêt pour l'apprentissage
%		- max_epoch: Nombre d'époque maximal autorisé
%
% Sorties:
%	* W: Cellule contenant les matrices de poids pour les L couches du perceptron
%	* b: Cellule contenant les biais pour les L couches du perceptron
%	* E: Valeur de la fonctionnelle d'erreur pour toutes les époques

% Définition de la fonction d'activation de la couche de sortie et de sa dérivée
gout = @(x) x;
gpout = @(x) ones(size(x));

% Définition de la fonction d'activation des couches cachées
g = @tanh;
gp = @(x) 1 - tanh(x).^2;

% Initialisation des paramètres généraux
eta = options.eta;
tol = options.tol;
max_epoch = options.max_epoch;
E = inf(max_epoch, 1);

% Initalisation du réseau
layers = [size(X, 1); hidden_layer(:); size(Y, 1)]; % Définition des différentes couches
L = numel(layers) - 1; % Nombre de couche du réseau (on omet la couche 0)

% Initialisation des poids et des biais selon la méthode dite de Xavier
W = cell(L, 1);
b = cell(L, 1);
for l = 1:L
    W{l} = sqrt(2/(layers(l) + layers(l+1)))*randn(layers(l),layers(l+1));
    b{l} = zeros(layers(l+1), 1);
end

% Initialisation de delta et de la valeur des neurones de sorties a
delta = cell(L, 1);
z = cell(L, 1);
a = cell(L+1, 1); % (a0, ..., aL)
a{1} = X;         % Initialisation de la couche d'entrée

% Apprentissage
for epoch = 2:max_epoch
	% 1. Propagation des poids et des bias de la couche d'entrée vers la couche de sortie
	for l = 1:L-1
		z{l} = W{l}.'*a{l} + b{l};
		a{l+1} = g(z{l}); % Couche cachées
	end
	z{L} = W{L}.'*a{L} + b{L};
	a{L+1} = gout(z{L});  % Couche de sortie
	
	% Contrôle de la convergence de l'apprentissage
	err = a{L+1} - Y;
	E(epoch) = dot(err(:), err(:))/2; % Valeur de la fonctionnelle d'erreur à l'époque 
								      % considérée
								  
	crit = abs(E(epoch) - E(epoch - 1))/E(epoch - 1);
	if crit <= tol
		break;
	end
	
	% 2. Rétropagation de la fonctionnelle d'erreur et mise à jours W et b
	for l = L:-1:1
		% Calcul de delta
		if l == L
			delta{l} = err.*gpout(z{l});
		else
			delta{l} = gp(z{l}).*(W{l+1}*delta{l+1});
		end
		
		% Calcul des gradients
		dW = a{l}*delta{l}.';
		db = sum(delta{l}, 2);
		
		% Mise à jour des poids et des biais
		W{l} = W{l} - eta*dW;
		b{l} = b{l} - eta*db;
	end
end
E = E(2:epoch);
```

## 5. Implémentation prédiction

```matlab
function Y = Pred_MultiLayer_Perceptron(W, b, X)
% Entrées
%	* W: Cellule contenant les matrices de poids pour les L couches du perceptron
%	* b: Cellule contenant les biais pour les L couches du perceptron
%	* X: Vecteur des entrées - dimensions (Nin x N)
%		 Nin: Nombre de paramètres d'entrée
%		 N: Nombre de valeurs considérées
%
% Sortie
%	* Y: Vecteur des sorties - dimensions (Nout x N)
%		 Nout: Nombre de paramètres de sortie

% Initalisation
L = length(W); % Nombre de couche hormis la couche 0
g = @tanh;
gout = @(x) x;

a = X;
for l = 1:L-1
	z = W{l}.'*a + b{l};
	a = g(z);
end
zout = W{L}.'*a + b{L};
Y = gout(zout);
```

## 6. Compléments

Pour limiter l'effet de sur-apprentissage, il peut être parfois intéressant de régulariser le calcul des poids. À cette fin, on considère la fonctionnelle suivante :
$$
\widetilde{E}(\mathbf{w}, \mathbf{b}) = E(\mathbf{w},\mathbf{b}) + \frac{\lambda}{2}\Vert\mathbf{w}\Vert_2^2
$$
 où $\lambda$ est le paramètre de régularisation.

En pratique, cela conduit à une modification du calcul des gradients de la fonctionnelle par rapport aux poids. Tous calculs faits, on obtient :
$$
\frac{\partial \widetilde{E}(\mathbf{w},\mathbf{b})}{\partial w_{ij}} = \frac{\partial E(\mathbf{w},\mathbf{b})}{\partial w_{ij}} + \lambda w_{ij}
$$
Du point de vue de l'implémentation, cela revient à modifier deux lignes de codes : elle relative au calcul de la fonctionnelle et celle relative au calcul du gradient. En pratique, on a donc :

```matlab
% Les lignes suivantes deviennent:
%    err = a{L+1} - Y;
%    L(epoch) = dot(err(:), err(:))/2;
err = a{L+1} - Y;
norm_w2 = sum(cellfun(@(x) dot(x(:),x(:)),W));
E(epoch) = dot(err(:), err(:))/2 + lambda*norm_w2/2;

% La ligne suivante devient:
% dW = a{l}*delta{l}.';
dW = a{l}*delta{l}.' + lambda*W{l};
```

Une autre manière d'éviter le sur-apprentissage consiste à stopper l'apprentissage de manière précoce. Une façon de déterminer le nombre d'époque optimal consiste à tracer, pour un ensemble de validation, la valeur de la fonctionnelle d'erreur en fonction du nombre d'époque et de choisir celui qui minimise la fonctionnelle.
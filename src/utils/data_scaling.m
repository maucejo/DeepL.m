function [Xscaled, varargout] = data_scaling(X, options, scale_from_another_scaling)

[Nfeatures, Nsamples] = size(X);
Xscaled = zeros(Nfeatures, Nsamples);
param = cell(1, Nfeatures);

if nargin == 1
    options = cell(1, Nfeatures);
    for ee = 1:Nfeatures
        options{ee}.type = 'minmax';
        options{ee}.range = [-1, 1];
    end
    scale_from_another_scaling = false;
elseif nargin == 2
    scale_from_another_scaling = false;
end


for ee = 1:Nfeatures
    Xp = X(ee, :);
    type = options{ee}.type;
    switch type
        case 'minmax'
            if scale_from_another_scaling
                xmin = options{ee}.xmin;
                xmax = options{ee}.xmax;
                a = options{ee}.a;
                b = options{ee}.b;
            else
                xmin = min(Xp);
                xmax = max(Xp);
                a = options{ee}.range(1);
                b = options{ee}.range(2);
                
                param{ee}.type = 'minmax';
                param{ee}.xmin = xmin;
                param{ee}.xmax = xmax;
                param{ee}.a = a;
                param{ee}.b = b;
            end
            
            Xscaled(ee, :) = a + (b - a)*(Xp - xmin)/(xmax - xmin);
            
        case 'meannorm'
            
            if scale_from_another_scaling
                xmean = options{ee}.xmean;
                xmin = options{ee}.xmin;
                xmax = options{ee}.xmax;
                
            else
                xmean = mean(Xp);
                xmin = min(Xp);
                xmax = max(Xp);
                param{ee}.type = 'meannorm';
                param{ee}.xmean = xmean;
                param{ee}.xmin = xmin;
                param{ee}.xmax = xmax;
            end
            
            Xscaled(ee, :) = (Xp - xmean)/(xmax - xmin);
            
        case 'standard'
            if scale_from_another_scaling
                xmean = options{ee}.xmean;
                xsdt = options{ee}.xsdt;
            else
                xmean = mean(Xp);
                xsdt = std(Xp);
                
                param{ee}.type = 'standard';
                param{ee}.xmean = xmean;
                param{ee}.xsdt = xsdt;
            end
            
            Xscaled(ee, :) = (Xp - xmean)/xsdt;
            
        case 'unitlength'
            if scale_from_another_scaling
                xnorm = options{ee}.xnorm;
                
            else
                xnorm = norm(Xp);
                param{ee}.type = 'unitlength';
                param{ee}.xnorm = xnorm;
            end
            
            Xscaled(ee, :) = Xp/xnorm;

    end
end

if ~scale_from_another_scaling
    varargout{1} = param;
end

function X = data_inverse_scaling(Xscaled, param)

[Nfeatures, Nsamples] = size(Xscaled);
X = zeros(Nfeatures, Nsamples);

for ee = 1:Nfeatures
    Xp = Xscaled(ee, :);
    type = param{ee}.type;
    
    switch type
        case 'minmax'
            xmin = param{ee}.xmin;
            xmax = param{ee}.xmax;
            a = param{ee}.a;
            b = param{ee}.b;
            
            X(ee, :) = xmin + ((Xp - a)*(xmax - xmin)/(b - a));
            
        case 'meannorm'
            xmean = param{ee}.xmean;
            xmin = param{ee}.xmin;
            xmax = param{ee}.xmax;
            
            X(ee, :) = xmean + (Xp*(xmax - xmin));
            
        case 'standard'
            xmean = param{ee}.xmean;
            xsdt = param{ee}.xsdt;
            
            X(ee, :) = xmean + xsdt*Xp;
            
        case 'unitlength'
            xnorm = param{ee}.xnorm;
            
            X(ee, :) = xnorm*Xp;
    end
end
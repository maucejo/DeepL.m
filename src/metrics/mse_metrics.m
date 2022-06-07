function mse = mse_metrics(Yout, Y)

Nsamples = size(Y, 2);

err = Yout - Y;
mse = dot(err(:), err(:))/Nsamples;
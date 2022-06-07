function rmse = rmse_metrics(Yout, Y)

Nsamples = size(Y, 2);

err = Yout - Y;
rmse = sqrt(dot(err(:), err(:))/Nsamples);
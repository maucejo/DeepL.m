function mae = mae_metrics(Yout, Y)

Nsamples = size(Y, 2);

err = Yout - Y;
mae = sum(abs(err), 'all')/Nsamples;
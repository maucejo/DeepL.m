function [loss, grad_loss] = mse_loss(Yout, Y)

Nsamples = size(Y, 2);

err = Yout - Y;
grad_loss = 2*err/Nsamples;
loss = dot(err(:), err(:))/Nsamples;

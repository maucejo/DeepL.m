function [loss, grad_loss] = sse_loss(Yout, Y)

grad_loss = Yout - Y;

loss = dot(grad_loss(:), grad_loss(:))/2;
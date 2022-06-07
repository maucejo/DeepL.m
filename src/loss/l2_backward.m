function l2_backward(layer, lambda)

if isprop(layer, 'W')
       layer.grad.weight = layer.grad.weight + lambda*layer.W;
end
function l1_backward(layer, lambda)

if isprop(layer, 'W')
    layer.grad.weight = layer.grad.weight + lambda*sign(layer.W);
end

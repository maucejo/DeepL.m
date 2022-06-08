function l1_l2_backward(layer, lambda)

if isprop(layer, 'W')
       layer.grad.weight = layer.grad.weight + lambda(1)*sign(layer.W) + lambda(2)*layer.W;
end
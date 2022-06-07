function reg_loss = l2_reg_loss(layers, lambda)

Nlayer = length(layers);

reg_loss = 0;
for ee = 1:Nlayer
   if isprop(layers{ee}, 'W')
       reg_loss = reg_loss + lambda*dot(layers{ee}.W(:), layers{ee}.W(:))/2;
   end
end
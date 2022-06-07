function reg_loss = l1_reg_loss(layers, lambda)

Nlayer = length(layers);

reg_loss = 0;
for ee = 1:Nlayer
   if isprop(layers{ee}, 'W')
       reg_loss = reg_loss + lambda*sum(abs(layers{ee}.W));
   end
end
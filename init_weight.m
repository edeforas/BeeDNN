function net=init_weight(net)

%init_weight as in http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

  nlayer=size(net.layer)(2);
  
  for i=1:nlayer
    layer=net.layer{i};
	  w=layer.weight;
    fan_in=columns(w);
    net.layer{i}.weight=((rand(rows(w),fan_in)-0.5)*2)/sqrt(fan_in);
  end

end

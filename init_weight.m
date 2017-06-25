function net=init_weight(net)

%init_weight as in http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

  nlayer=size(net.layer)(2);
  
  for i=1:nlayer
    layer=net.layer{i};
	  w=layer.weight;
    fan_in=columns(w);
    fan_out=rows(w);
    r=sqrt(6/(fan_in+fan_out));
    
    if strcmp(layer.func,'sigmoid')
      r=r*4;
    end %todo add more function special case
    
    net.layer{i}.weight=((rand(fan_out,fan_in)-0.5)*2)*r;    
  end

end

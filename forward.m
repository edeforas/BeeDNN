function out=forward(net,data)
  nlayer=size(net.layer)(2);
  
  out=data;
  
  for i=1:nlayer
    layer=net.layer{i};
    
    out=layer.weight*out;
    out=activation(layer.func,out);
  end
end

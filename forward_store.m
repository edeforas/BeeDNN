function [out,netout]=forward_store(net,data)
  
  netout=net;
  nlayer=size(net.layer)(2);
  
  out=data;
  
  for i=1:nlayer
    layer=net.layer{i};
    
    netout.layer{i}.in=[out;1];
    outweight=layer.weight*out;
    out=activation(layer.func,outweight);

    netout.layer{i}.out=out;
    netout.layer{i}.outweight=outweight;
    
  end
end
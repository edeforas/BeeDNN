function layer=create_layer(in_size,out_size,func)
  layer.weight=rand(in_size+1,out_size)-0.5; %todo init with fan_in , fan_out, func type
  layer.func=func;
end  
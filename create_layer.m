function layer=create_layer(in_size,out_size,func)
  layer.weight=rand(out_size,in_size+1)-0.5; %todo init wiht fan_in , fan_out, func type
  layer.func=func;
end  
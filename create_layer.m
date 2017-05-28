function layer=create_layer(in_size,out_size,func)
  layer.weight=rand(out_size,in_size)-0.5;
  layer.func=func;
end  
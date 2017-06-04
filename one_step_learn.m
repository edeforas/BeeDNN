function [net,error]=one_step_learn(netin,in,out_truth)
  
  [out,net]=forward_store(netin,in);
  error=out-out_truth;
  
  learning_rate=net.learning_rate;
  nlayer=size(net.layer)(2);
 
  for i=nlayer:-1:1
	
    layer=net.layer{i};
   % weight_layer=layer.weight;
    func=layer.func;
    outweight=layer.outweight;

    if i==nlayer
       % output layer
       delta=error.*activation_derivation(layer.func,outweight);
    else
       % hidden layer
       a=  (net.layer{i+1}.weight') *  delta; % use of previous delta
       a=a(1:rows(a)-1,:); % do not use last weight (use only for bias)
       b=activation_derivation(layer.func,outweight); 
       delta=a.*b;
    end
    
    dE=delta*(layer.in');
    
    % momentum as in: http://distill.pub/2017/momentum/
    if isfield(layer,"last_dE")
      dE=dE+net.momentum*layer.last_dE;
    end
    net.layer{i}.last_dE=dE;
    
    net.layer{i}.weight=net.layer{i}.weight-learning_rate*dE;
 
  end
  
end

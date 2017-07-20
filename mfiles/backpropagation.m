function [net,error]=backpropagation(net,error) %todo use truth and last out stored in net?
  
  %[out,net]=forward_store(netin,in);
  %error=out-out_truth;
  
  learning_rate=net.learning_rate;
  nlayer=size(net.layer)(2);
 
  for i=nlayer:-1:1
	
    layer=net.layer{i};
    func=layer.func;
    outweight=layer.outweight;

    if i==nlayer
       % output layer
       delta=error.*activation_derivation(layer.func,outweight);
    else
       % hidden layer
       a=  delta*(net.layer{i+1}.weight') ;%*  delta; % use of previous delta
%       a=a(1:rows(a)-1,:); % do not use last weight (use only for bias)
       a=a(:,1:columns(a)-1); % do not use last weight (use only for bias)
       b=activation_derivation(layer.func,outweight); 
       delta=a.*b;
    end
    
    dE=(layer.in')*delta;
    net.layer{i}.dE=dE;
 %   net.layer{i}.weight=net.layer{i}.weight-learning_rate*dE;
 
  end  
end
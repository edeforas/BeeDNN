
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out=activation(func,data)
  out=data;
  
  if(strcmp(func,'relu'))
    out(out<0)=0;
  end
  
  if(strcmp(func,'sigmoid'))
    out=1./(1.+exp(-out));
  end

  if(strcmp(func,'softplus'))
    out=log(1+exp(out));
  end

  if(strcmp(func,'tanh'))
    out=tanh(out);
  end
  
  if(strcmp(func,'elliot'))
    out=0.5 .* (out ./ (1 + abs(out))) + 0.5;
  end
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


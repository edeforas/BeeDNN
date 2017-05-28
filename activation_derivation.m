%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out=activation_derivation(func,data)
  out=data;
  
  if(strcmp(func,'relu'))
    out(out<0)=0;
    out(out>0)=1;
  end
  
  if(strcmp(func,'sigmoid'))
    s=1./(1.+exp(-out));
    out=s.*(1-s);
  end

  if(strcmp(func,'softplus'))
    out=1./(1.+exp(-out));
  end

  if(strcmp(func,'tanh'))
    t=tanh(out);
    out=1-t.*t;
  end
  
  if(strcmp(func,'elliot'))
    out=0.5 ./((1 + abs(out)).*(1 + abs(out)));
  end
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

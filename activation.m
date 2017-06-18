
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
  
  if(strcmp(func,'atan'))
    out=atan(out);
  end

  if(strcmp(func,'softsign'))
    out=out./(1+abs(out));
  end

  if(strcmp(func,'gauss'))
    out=exp(-out.*out);
  end

  if(strcmp(func,'elu'))
    alpha=1;
    outn=out(out<0);
    out(out<0)=alpha.*(exp(outn)-1);
  end

  if(strcmp(func,'linear'))
    % nothing to do
  end
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


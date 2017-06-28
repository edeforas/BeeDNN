
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y=activation(func,x)

  if(strcmp(func,'relu'))
  y=x;
  y(y<0)=0;
  end
  
  if(strcmp(func,'sigmoid'))
  y=1./(1.+exp(-x));
  end

  if(strcmp(func,'softplus'))
  y=log(1+exp(x));
  end

  if(strcmp(func,'tanh'))
  y=tanh(x);
  end
  
  if(strcmp(func,'elliot'))
  y=0.5 .* (x ./ (1 + abs(x))) + 0.5;
  end
  
  if(strcmp(func,'atan'))
  y=atan(x);
  end

  if(strcmp(func,'softsign'))
  y=x./(1+abs(x));
  end

  if(strcmp(func,'gauss'))
  y=exp(-x.*x);
  end

  if(strcmp(func,'elu'))
%    alpha=1;
%  out=data; TODO
%  outn=out(out<0);
%    out(out<0)=alpha.*(exp(outn)-1);
  end

  if(strcmp(func,'selu'))
  lambda = 1.05070;
  alpha  = 1.67326;
  
  s=x;
  sn=s(s<0);
  s(x<0)=alpha*(exp(sn)-1);
  y=lambda*s;
  end

  if(strcmp(func,'linear'))
  y=x;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out=activation_derivation(func,x)
  
  if(strcmp(func,'relu'))
    out=x;
    out(out<0)=0;
    out(out>0)=1;
  end
  
  if(strcmp(func,'sigmoid'))
    s=1./(1.+exp(-x));
    out=s.*(1-s);
  end

  if(strcmp(func,'softplus'))
    out=1./(1.+exp(-x));
  end

  if(strcmp(func,'tanh'))
    t=tanh(x);
    out=1-t.*t;
  end
  
  if(strcmp(func,'elliot'))
    out=1+abs(x);
    out=0.5 ./(out.*out);
  end
  
  if(strcmp(func,'atan'))
    out=1./(1+x.*x);
  end
  
  if(strcmp(func,'softsign'))
    out=1+abs(x);
    out=1./(out.*out);
  end

  if(strcmp(func,'gauss'))
    out=-2.*x.*exp(-x.*x);
  end

  if(strcmp(func,'selu'))
    lambda = 1.05070;
  alpha  = 1.67326;
  
  s=x;
  s(s>=0)=1;  
  sn=s(s<0);
  s(x<0)=alpha*(exp(sn));

  out=lambda*s;
  end

  
  if(strcmp(func,'linear'))
    out=x*0+1;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

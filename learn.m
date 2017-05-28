function [netout,error]=learn(net,samples,truth,nbiter)
  
  netout=net;
  nbiter=net.nbiter;
  stoperror=net.stoperror;
  
  error=[];
  for i=1:nbiter
    max_error=0;
    for u=1:columns(samples) %todo shuffle
      [netout,err]=one_step_learn(netout,samples(:,u),truth(u));
      max_error=max(max_error,abs(err));
    end 
    error=[error, max_error];
   
   if(max_error<stoperror)
     return;
   end;
 end
end
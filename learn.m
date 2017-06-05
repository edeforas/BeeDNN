function [net,error]=learn(net,samples,truth,nbiter)
  nbiter=net.nbiter;
  stoperror=net.stoperror;
  
  nbsamples=columns(samples);
  error=[];
  for i=1:nbiter
  
    %randomly permute samples
    idxperm=randperm(nbsamples);

    max_error=0;
    
    for u=1:nbsamples % todo add batch and mini batch
  		[out,net]=forward_feed(net,samples(:,idxperm(u)));
      err=out-truth(:,idxperm(u));
      
      net=backward(net,err);

		  max_error=max(max_error,abs(err));
    end
    error=[error, max_error];
   
   if(max_error<stoperror)
     return;
   end;
 end
end
function [net,error]=learn(net,samples,truth,nbiter)
  nbiter=net.nbiter;
  stoperror=net.stoperror;
  
  nbsamples=columns(samples);
  
  if(isfield(net,"batch_size"))
    batch_size= net.batch_size;
  else
    batch_size=1; % SGD case
  end
  
  nb_batches=ceil(nbsamples/batch_size);
  
  %error=[];
  for i=1:nbiter
  
    %randomly permute samples
    idxperm=randperm(nbsamples);

    batch_start=1;
    for ib=1:nb_batches
      batch_end=min(batch_start+batch_size,nbsamples);
      
      netfull=net;

      sample_size=batch_end-batch_start;
      for u=batch_start:batch_end
        	[out,netout]=forward_feed(net,samples_batch(:,idxperm(u)));
          err=out-truth(:,idxperm(u)); 
          netfeed=backward(netout,err);

        % get and mean dE
        % add in netfull
        
      end
      
      %apply dE/sample_size from netfull
      
      batch_start=batch_start+batch_size;
    end
    
    
    
    
    
    
    max_error=0;
    
    for u=1:nbsamples % todo add batch and mini batch
  		[out,net]=forward_feed(net,samples(:,idxperm(u)));

      
    

		  max_error=max(max_error,abs(err));
    end
    error=[error, max_error];
   
   if(max_error<stoperror)
     return;
   end;
 end
end
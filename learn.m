function [net,error]=learn(net,samples,truth)

  net=init_weight(net);

  stoperror=net.stoperror;
  nlayer=size(net.layer)(2);
 
  if(isfield(net,"batch_size"))
    batch_size=net.batch_size;
  else
      batch_size=1;
  end
  
  nbsamples=rows(samples);
  nb_batch=ceil(nbsamples/batch_size);

  % init dE
  for i=1:nlayer
		net.layer{i}.dE=net.layer{i}.weight*0;    
	end
  
  error=[];
  for i=1:net.epochs
  
    %randomly permute samples
    idxperm=randperm(nbsamples);

    max_error=0;

    start_batch=1;
    for i=1:nb_batch;
      end_batch=min(start_batch+batch_size-1,nbsamples);
      
      sample_batch=samples(idxperm(start_batch:end_batch),:);
      truth_batch=truth(idxperm(start_batch:end_batch),:); 
    
      max_batch_error=0;
      
	    %init netaccum
	    netaccum=net;
	    for i=1:nlayer
		    netaccum.layer{i}.dE=netaccum.layer{i}.weight*0;    
	    end
	  
      % iterate over all samples in batch
	    for u=1:rows(sample_batch)
  		  [out,netbatch]=forward_feed(net,sample_batch(u,:));
        err=(out-truth_batch(u,:));
        netbatch=backpropagation(netbatch,err);
  	    max_batch_error=max(abs(err),max_batch_error);

        %add dE from net to netaccum
		    for i=1:nlayer
			    netaccum.layer{i}.dE=netaccum.layer{i}.dE+netbatch.layer{i}.dE;    
		    end
	  
      end
    	  
      % compute average, apply momentum, update weight
	    for i=1:nlayer
		 %   netaccum.layer{i}.dE=netaccum.layer{i}.dE/columns(nbsamples);   % do we divide by columns(nbsamples) ?
		    net.layer{i}.dE=netaccum.layer{i}.dE+net.momentum*net.layer{i}.dE;
		    net.layer{i}.weight=net.layer{i}.weight-net.learning_rate*net.layer{i}.dE;
	    end
	       
      %update error
	    max_error=max(max_error,max_batch_error);
    
      start_batch=start_batch+batch_size;
    end
  
    error=[error, max_error];
   
    if(max_error<stoperror)
      return;
    end;
  end
end
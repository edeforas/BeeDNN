function [net,error]=learn_batch(net,samples,truth,nbiter)
  nbiter=net.nbiter;
  stoperror=net.stoperror;
    nlayer=size(net.layer)(2);
  
  batch_size=1;
  if(isfield(net,"batch_size"))
    batch_size=net.batch_size;
  end
  
  nbsamples=columns(samples);
  nb_batch=ceil(nbsamples/batch_size);

  error=[];
  for i=1:nbiter
  
    %randomly permute samples
    idxperm=randperm(nbsamples);

    max_error=0;

    start_batch=1;
    for i=1:nb_batch;
      end_batch=min(start_batch+batch_size,nbsamples);
      
      sample_batch=samples(:,idxperm(start_batch:end_batch));
      truth_batch=truth(:,idxperm(start_batch:end_batch));
    
      max_batch_error=0;
      
	    %init netaccum
	    netaccum=net;
	    for i=1:nlayer
		    netaccum.layer{i}.dE=netaccum.layer{i}.weight*0;    
	    end
	  
	    for u=1:columns(sample_batch)
  		  [out,netbatch]=forward_feed(net,sample_batch(:,u));
        err=out-truth_batch(:,u);
        net=backpropagation(netbatch,err);
  	    max_batch_error=max(err,max_batch_error);

        %add dE from net
		    for i=1:nlayer
			    netaccum.layer{i}.dE=netaccum.layer{i}.dE+net.layer{i}.dE;    
		    end
	  
      end
    	  
      % compute average, apply momentum, update weight
	    for i=1:nlayer
		    netaccum.layer{i}.dE=netaccum.layer{i}.dE/columns(sample_batch);   
		    net.layer{i}.dE=netaccum.layer{i}.dE+net.momentum*net.layer{i}.dE;
		    net.layer{i}.weight=net.layer{i}.weight-net.learning_rate*net.layer{i}.dE;
	    end
	       
      %update error
	    max_error=max(max_error,abs(max_batch_error));
    
      start_batch=start_batch+nbsamples;
    end
  
    error=[error, max_error];
   
    if(max_error<stoperror)
      return;
    end;
  end
end
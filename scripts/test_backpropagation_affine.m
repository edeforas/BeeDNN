% this script implement the backpropagation method of a affine fit , from scratch, WIP

input= [ -2 3 -1 4 1 2];
target=2*input+1; %create reference: samples*2+1
learning_rate=0.05;
nb_epochs=100;
AInit=8; #gain model A*x+B ; initial starting point
BInit=5; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out=forward_relu(in,W)
  out=max(in,0);
end
function [newW,newDelta]=backpropagation_relu(in,learning_rate,delta,W)

   % update input error
  if(in>0)
    newDelta=delta;
  else
    newDelta=0;
  endif
  
   % update internal parameter
   newW=0; % nothing to do
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out=forward_gain(in,W)
  out=in*W;
end
function [newW,newDelta]=backpropagation_gain(in,learning_rate,delta,W)

   % update input error
   newDelta=delta*in;
   
   % update internal parameter
   newW=W-delta*learning_rate*in;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out=forward_bias(in,W)
  out=in+W;
end
function [newW,newDelta]=backpropagation_bias(in,learning_rate,delta,W)

   % update input error
   newDelta=delta;
   
   % update internal parameter
   newW=W-delta*learning_rate;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Atotal,Btotal]=learn(input,target,learning_rate,nb_epochs,nb_samples_minibatch,AInit,BInit)
A=AInit;
B=BInit;
Atotal=[AInit];
Btotal=[BInit];

nb_samples=columns(input);
nb_batch=floor(nb_samples/nb_samples_minibatch);
for epochs=1:nb_epochs
  idx=randperm(nb_samples);
  in_shuffled=input(idx);
  out_shuffled=target(idx);
       
    ABatch=0;
    BBatch=0;
  for b=1:nb_batch
    InTotal=[];
    OutTotal=[];
    error=0;
      in2=0;
      
      %mini batch forward pass
    for i=1:nb_samples_minibatch
      in=in_shuffled(i+(b-1)*nb_samples_minibatch);
      out=forward_gain(in,A);
      in2+=out;
      out=forward_bias(out,B);
      InTotal=[InTotal;in];
      OutTotal=[OutTotal; out];
      error+=out-out_shuffled(i+(b-1)*nb_samples_minibatch);            
    end
    
    %update rule for A using error
    error/=nb_samples_minibatch;
    in2/=nb_samples_minibatch;
    inmean=mean(InTotal,1);
    
    [B,delta]=backpropagation_bias(in2,learning_rate,error,B);
    [A,delta]=backpropagation_gain(inmean,learning_rate,delta,A);
    
    ABatch+=A;
    BBatch+=B;
  end
  
  Atotal=[Atotal,ABatch/nb_batch];
  Btotal=[Btotal,BBatch/nb_batch];
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%compute results
[ABatch,BBatch]=learn(input,target,learning_rate,nb_epochs,6,AInit,BInit);
[AMiniBatch,BMiniBatch]=learn(input,target,learning_rate,nb_epochs,3,AInit,BInit);
[AMicroBatch,BMicroBatch]=learn(input,target,learning_rate,nb_epochs,2,AInit,BInit);
[ASGD,BSGD]=learn(input,target,learning_rate,nb_epochs,1,AInit,BInit);

%compute error landscape
errorLandscape=[];
for a=-1:0.1:9
errb=[];
for b=-1:0.1:9
  err=0;
for s=1:6
  delta=forward(input(s),a,b)-target(s);
  err+=delta*delta/2;
end
errb=[errb; err/6];
end
errorLandscape=[errorLandscape errb];
end

%draw convergence curve in error landscape
figure; hold on;
imagesc([-1 9],[-1 9],errorLandscape);
plot(ABatch,BBatch,';batch;k');
plot(AMiniBatch,BMiniBatch,';minibatch;r');
plot(AMicroBatch,BMicroBatch,';microbatch;g');
plot(ASGD,BSGD,';SGD;b');
colormap(jet);
grid;
hold off;

%draw A,B vs epoch
figure; hold on;
plot(ABatch,';batch;k');
plot(BBatch,';batch;+k');
plot(AMiniBatch,';minibatch;r');
plot(BMiniBatch,';minibatch;+r');
plot(AMicroBatch,';microbatch;g');
plot(BMicroBatch,';microbatch;+g');
plot(ASGD,';SGD;b');
plot(BSGD,';SGD;+b');
hold off;

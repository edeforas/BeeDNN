%simple NN backpropagation on 1D sin regression

disp 'please wait, 1000 epochs ...';

clear net;
net.layer{1}=create_layer(1,20,'sigmoid');
net.layer{2}=create_layer(20,20,'sigmoid');
net.layer{3}=create_layer(20,1,'tanh');
net.learning_rate=0.1;
net.momentum=0.2;
net.epochs=1000;
net.stoperror=0.05;
net.batch_size=10;

samples=0:0.1:6.3;
truth=sin(samples);

[net,error]=learn(net,samples,truth);

figure;
plot(error), title('Network loss');
xlabel('Iteration'), ylabel('Loss');

%show result
j=0:0.01:6.3;
v=[];
for i=j
    v=[v,forward(net,i)];
end
  
figure; hold on;
plot(j,v,'r;DNN_result;');
s=sin(j);
plot(j,s,'b;Truth;');
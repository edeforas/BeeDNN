%simple NN backpropagation on 1D sin regression

disp 'please wait, 10000 iterations...';

clear net;

net.layer{1}=create_layer(1,10,'sigmoid');
net.layer{2}=create_layer(10,10,'sigmoid');
net.layer{3}=create_layer(10,1,'tanh');
net.learning_rate=0.1;
net.momentum=0.1;
net.nbiter=10000;
net.stoperror=0.05;

samples=0:0.5:10;
truth=sin(samples);

[net,error]=learn(net,samples,truth);

figure;
plot(error), title('Network loss');
xlabel('Iteration'), ylabel('Loss');

%show result
v=[];
for i=0:0.1:10
    v=[v,forward(net,i)];
end
  
figure; hold on;
j=0:0.1:10;
plot(j,v,'r');
s=sin(j);
plot(j,s,'b');
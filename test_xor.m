%simple NN backpropagation on 2D xor function return 1 if a!=b 0 otherwise

clear net;

net.layer{1}=create_layer(2,3,'sigmoid');
net.layer{2}=create_layer(3,1,'sigmoid');
net.learning_rate=2;
net.momentum=0.9;
net.nbiter=500;
net.stoperror=0.05;

samples=[0 0 1 1;  ...
         0 1 0 1];
         
truth=[0 1 1 0];

[net,error]=learn(net,samples,truth);

%show intermediate result on a 2d map
%u=[];
%for j=0:0.1:1
%  v=[];
%  for i=0:0.1:1
%      v=[v,forward(net,[i; j])];
%  end
%  u=[v;u];
%end
%figure; imagesc(u);
%round(u*10)

plot(error), title('Network loss');
xlabel('Iteration'), ylabel('Loss');

%test with binary values
forward(net,[0;0])
forward(net,[1;0])
forward(net,[0;1])
forward(net,[1;1])
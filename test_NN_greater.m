%simple NN backpropagation on 2D greater function return 1 if a>b 0 otherwise

clear net;

net.layer{1}=create_layer(2,1,'sigmoid');
net.learning_rate=4;
net.momentum=0.9;
net.nbiter=1000;
net.stoperror=0.01;

samples=[0 0 1 1; 0 1 0 1];  % todo transpose samples
truth=[0.5 0 1 0.5];

[net,error]=learn(net,samples,truth);

%show result
u=[];
for j=0:0.1:1
  v=[];
  for i=0:0.1:1
      v=[v,forward(net,[i; j])];
  end
  u=[v;u];
end
%figure; imagesc(u);

plot(error);

round(u*10)

forward(net,[0;0])
forward(net,[1;0])
forward(net,[0;1])
forward(net,[1;1])
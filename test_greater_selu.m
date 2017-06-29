%simple NN backpropagation on 2D greater function return 1 if a>b 0.5 if equal , 0 otherwise

clear net;

net.layer{1}=create_layer(2,2,'selu');
net.layer{2}=create_layer(2,1,'sigmoid');
net.learning_rate=0.05;
net.momentum=0.5;
net.epochs=500;
net.stoperror=0.05;
net.batch_size=4;

samples=[0 0 1 1; ...
         0 1 0 1];
         
truth=[0.5 0 1 0.5];

[net,error]=learn(net,samples,truth);

plot(error), title('Network loss');
xlabel('Iteration'), ylabel('Loss');

forward(net,[0;0])
forward(net,[1;0])
forward(net,[0;1])
forward(net,[1;1])
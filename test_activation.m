%simple activation function and derivate, draw

x=-5:0.1:5;

%plot activation functions
sigmoid_x=activation('sigmoid',x);
tanh_x=activation('tanh',x);
relu_x=activation('relu',x);
softplus_x=activation('softplus',x);
elliot_x=activation('elliot',x);
atan_x=activation('atan',x);
softsign_x=activation('softsign',x);
gauss_x=activation('gauss',x);
linear_x=activation('linear',x);

figure; hold on;
plot(x,sigmoid_x,'r;sigmoid;');
plot(x,tanh_x,'b;tanh;');
plot(x,relu_x,'b;relu;');
plot(x,softplus_x,'k;softplus;');
plot(x,elliot_x,'g;elliot;');
plot(x,atan_x,'g;atan;');
plot(x,softsign_x,'g;softsign;');
plot(x,gauss_x,'g;gauss;');
plot(x,linear_x,'g;linear;');

title('activation function');

%plot activation functions derivates
sigmoid_deriv_x=activation_derivation('sigmoid',x);
tanh_deriv_x=activation_derivation('tanh',x);
relu_deriv_x=activation_derivation('relu',x);
softplus_deriv_x=activation_derivation('softplus',x);
elliot_deriv_x=activation_derivation('elliot',x);
atan_deriv_x=activation_derivation('atan',x);
softsign_deriv_x=activation_derivation('softsign',x);
gauss_deriv_x=activation_derivation('gauss',x);
linear_deriv_x=activation_derivation('linear',x);

figure; hold on;
plot(x,sigmoid_deriv_x,'r;sigmoid deriv;');
plot(x,tanh_deriv_x,'b;tanh deriv;');
plot(x,relu_deriv_x,'b;relu deriv;');
plot(x,softplus_deriv_x,'k;softplus deriv;');
plot(x,elliot_deriv_x,'g;elliot deriv;');
plot(x,atan_deriv_x,'g;atan deriv;');
plot(x,softsign_deriv_x,'g;softsign deriv;');
plot(x,gauss_deriv_x,'g;gauss deriv;');
plot(x,linear_deriv_x,'g;linear deriv;');

title('activation derivate function');

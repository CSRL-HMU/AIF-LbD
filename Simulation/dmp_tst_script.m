close all
clear all



dt = 0.002;


t=[0:dt:10];
N = length(t);


for i=1:N
    retVal = get5thOrder(t(i), 0.5, -0.3, 9.0);
    pd(i) = retVal(1,1);
    pd_dot(i) = retVal(1,2);
    pd_ddot(i) = retVal(1,3);
end


dmp_model = dmp(10, t(end), 'gaussian' , 'linear', 4, 40, 1, 1);

pd_noise = pd+rand(1,N)*0.1;

dmp_model = dmp_model.train(dt, pd_noise, true, false, 1.0);




p0 = -0.7
state = [0 ; p0 ; 0];
dx = 0;
dy = 0;
dz = 0;

dmp_model = dmp_model.set_goal(0.2);
dmp_model = dmp_model.set_init_position(p0);

% dmp_model = dmp_model.set_tau(0.5);


for i =1:N
    x(:,i) = state;
    state = state + [dx;dy;dz]*dt;

  
    [dx,dy,dz] = dmp_model.get_state_dot(state(1), state(2), state(3), false, 1.0);



end


plot(t,pd_noise)
hold on
plot(t,x(2,:))
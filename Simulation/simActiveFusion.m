clear all
close all 

%% Inialize time for Simulation
dt = 0.001;
t=[0:dt:10];
N = length(t);

%% Initialize Pk (the covariance of the accumulated estimate)
Pk = zeros(N,2,2);
%Pk = diag(rand(5,1));

%% Create pd as the ground truth of the finger 
for i=1:N
    retVal = get5thOrder(t(i), 0.5, -0.3, 9.0);
    pd(:,i) = [0.5*cos(2*pi*0.1*t(i))+retVal(1,1)+0.2 ; -retVal(1,1)*1.3-0.2];

    % Initialize the values of Pk
    Pk(i,:,:) = eye(2,2)*100.0;
end


%% Set the parameters of the system
M = 40;
alpha = 4;
beta = 1;
W = zeros(M,2);
ka = 400.0;
sigma1 = 0.8;
sigma2 = 0.03;
Sigma0 = diag([sigma1^2 ,sigma2^2]);
enU = true;
online_figs =false;

%% Initial and target position for the DMP
pT = pd(:,end);
p0 = pd(:,1);

%% Initialize the DMP
dmpx = dmp(M, t(end), 'gaussian', 'linear', 4, alpha, beta, 1);
dmpy = dmp(M, t(end), 'gaussian', 'linear', 4, alpha, beta, 1);

pc(1,1) = mean(pd(1,:));
pc(2,1) = mean(pd(2,:));


for i=1:10 %% for each repetition

    close all
    noise = 0;
    

    %% Initialize the sensor position and orientation
    pp = [3;0]+enU*[0;(i-1)*0.1];
    thetap = atan2(pc(2)-pp(2), pc(1)-pp(1));
    thetap=thetap ;
    sensor_state = [pp ; thetap];
    vp = [0; 0; 0];
   
    d0 = norm(pc(:,1) - pp);

    %% Pass the previous P_k to Q_k+1
    Qk = Pk;

    
    %% Inialize the state of KALMAN model (DMP)
    xpred_k = [p0; 0; 0; 0];


    %% Initialize DMP parameters
    dmpx = dmpx.set_goal(pT(1));
    dmpy = dmpy.set_goal(pT(2));

    dmpx = dmpx.set_init_position(p0(1));
    dmpy = dmpy.set_init_position(p0(2));

    for k=1:N  %% for each discrete time k 

        %% Euler integration
        sensor_state = sensor_state + vp * dt;
        sensor_state_log(:,k) = sensor_state;

        %% Get position and orientation
        pp = sensor_state(1:2);
        thetap = sensor_state(3);

        % The x and y vectors of the sensor
        xp = [cos(thetap);sin(thetap)];
        yp = [cos(thetap+0.5*pi);sin(thetap+0.5*pi)];
        R0p = [xp yp];
        Rp0 = R0p';
        
        %% Data logging
        pp_log(:,k) = pp;
        theta_log(k) = thetap;
        
        %% simulate the POI position and uncertainty
        p_real = Rp0*(pd(:,k) - pp);
        Sigma = R0p*Sigma0*R0p';

        Sigma_log(k,:,:)= Sigma;
        
        %% Simulate the measurement noise
        epsilon_p = sqrt(Sigma0)*normrnd([0;0],1);
        % epsilon = R0p*epsilon;  

        %% Get measurement
        % w.r.t. the sensor frame
        % Simulate noise from t=4s to t=4.5s
        if(t(k)>3.0 && t(k)<3.5)
            if mod(k,40)==30
                noise = epsilon_p;
            end
        else
            noise = 0;
        end
        p_hatp = p_real + noise; 
        % w.r.t. the world frame 
        p_hat = pp + R0p*p_hatp;
        pmeas_log(:,k) = p_hat;

        %% Data fusion ----------------
        % Compute model prediction
        [xretv3, xretv1, xretv2] = dmpx.get_state_dot(xpred_k(5), xpred_k(1), xpred_k(3), false, 1);
        [yretv3, yretv1, yretv2] = dmpy.get_state_dot(xpred_k(5), xpred_k(2), xpred_k(4), false, 1);
    
        state_dot = [xretv1 ; yretv1 ; xretv2 ; yretv2 ; xretv3];
        xpred_k = xpred_k + state_dot*dt;

        % COVARIANCE WEIGHTED MEAN
        phatStar(:,k) =  inv(inv(squeeze(Qk(k,:,:)))+inv(Sigma))*(inv(squeeze(Qk(k,:,:)))*xpred_k(1:2) + inv(Sigma)*p_hat);


        %% Compute Pk
        Pk(k,:,:) = inv(inv(squeeze(Qk(k,:,:)))+inv(Sigma));
        VPk_log(k) = det(squeeze(Pk(k,:,:)));
       

        %% Active perception
        
        % This is an intermediate matrix for optimal calculations
        Pinv = inv(squeeze(Pk(k,:,:)));
        detP = det(squeeze(Pk(k,:,:)));
        S_inv = inv(Sigma);

        A =  squeeze(Pk(k,:,:)) * squeeze(Pk(k,:,:)) * S_inv * S_inv;
        %B = H *  P_kkprev ;

        % dR0p / dtheta  
        dR0p = [-sin(thetap) -cos(thetap); ....
                 cos(thetap) -sin(thetap) ];
        % dSigma / dtheta  
        dSigma_dtheta = dR0p * Sigma0 * R0p' + ....
                        R0p * Sigma0 * dR0p' ;
        % dV / dtheta 
        dVtheta = detP * trace(Pinv * A * dSigma_dtheta);

        % dV / dz (total gradient)
        dV = [0 ; 0 ; dVtheta];

    

        %% CONTROL SIGNAL !
        vp = - ka * eye(3,3) * dV;

        u = cross([xp;0],[pc-pp;0]); 

        vv = cross([d0*(pc-pp)/norm(pc-pp);0],[0;0;vp(3)]) ;
        vp(1:2) =  vp(1:2) +  vv(1:2);
        % vp(3) =  vp(3) + 4.0*u(3);
        
        vp = enU*vp;

        if(vp~=vp)
            input('opa')
        end
        % vp =0*vp;
        vp_log(:,k) = vp;
        detP_log(k) = detP;

        if(mod(k,100)==0 && online_figs)
            %%plots 
            figure(1)
            hold off
            plot(pmeas_log(1,1:k),pmeas_log(2,1:k),'g-')
            
            hold on
            plot(phatStar(1,1:k),phatStar(2,1:k))
            plot(pd(1,1:k),pd(2,1:k),LineWidth=2)
            plot(pd(1,1),pd(2,1),'x-',LineWidth=2,MarkerSize=5)
            plot(pd(1,k),pd(2,k),'o-',LineWidth=2,MarkerSize=5)
            plot(sensor_state_log(1,1:k),sensor_state_log(2,1:k),'r:')
               
                
            S_now = squeeze(Sigma_log(k,:,:));
           
            P_now = squeeze(Pk(k,:,:));
            Q_now = squeeze(Qk(k,:,:));
      
            inv(S_now)
            xyS = 1.0*plot_ellipse(inv(S_now));
            xyP = 1.0*plot_ellipse(inv(P_now));
            xyQ = 1.0*plot_ellipse(inv(Q_now));

        
            plot(phatStar(1,k) + xyS(1,:),phatStar(2,k) + xyS(2,:),'k');
            plot(phatStar(1,k) + xyP(1,:),phatStar(2,k) + xyP(2,:),'m');
            plot(phatStar(1,k) + xyQ(1,:),phatStar(2,k) + xyQ(2,:),'g');

            pp = sensor_state_log(1:2,k);
            thetap = sensor_state_log(3,k);
            xp = [cos(thetap) ; sin(thetap)];
            yp = [cos(thetap+pi/2) ; sin(thetap+pi/2)];
            
            plot([pp(1) pp(1)+0.5*xp(1)],[pp(2) pp(2)+0.5*xp(2)],'go-',MarkerSize=3,LineWidth=1)
            plot(pp(1),pp(2),'r+',MarkerSize=4,LineWidth=3)
            
           
            axis equal
            % pause(0.05)
        end


        
    end
    p0 = phatStar(:,1);
    pT = phatStar(:,end);
    
    %% Train both DMPs
    dmpx = dmpx.train(dt, phatStar(1,:), true, false, 1.0);
    dmpy = dmpy.train(dt, phatStar(2,:), true, false, 1.0);


    
    xst = [0;p0(1);0];
    yst = [0;p0(2);0];

    dmpx = dmpx.set_init_position(p0(1));
    dmpy = dmpy.set_init_position(p0(2));

    dmpx = dmpx.set_goal(pT(1));
    dmpy = dmpy.set_goal(pT(2));

    for j=1:N
        [dax, dbx, dcx] = dmpx.get_state_dot(xst(1), xst(2), xst(3), false, 1);
        [day, dby, dcy] = dmpy.get_state_dot(yst(1), yst(2), yst(3), false, 1);
        xst = xst + [dax; dbx; dcx]*dt;
        yst = yst + [day; dby; dcy]*dt;

        xst_log(:,j) =xst; 
        yst_log(:,j) =yst; 
 
        
    end

    figure(100)
    plot(t,xst_log(2,:),'b-')
    hold on
    plot(t,pd(1,:),'b--')
    plot(t,yst_log(2,:),'r-')
    plot(t,pd(2,:),'r--')
    

    figure()
    plot(pmeas_log(1,:),pmeas_log(2,:),'g-')
    
    hold on
    plot(phatStar(1,:),phatStar(2,:))
    plot(pd(1,:),pd(2,:),LineWidth=2)
    plot(pd(1,1),pd(2,1),'x-',LineWidth=2,MarkerSize=5)
    plot(pd(1,end),pd(2,end),'o-',LineWidth=2,MarkerSize=5)
    plot(sensor_state_log(1,:),sensor_state_log(2,:),'r:')
    
    for j=1:4000:N
        
        S_now = squeeze(Sigma_log(j,:,:));
        P_now = squeeze(Pk(j,:,:));
        Q_now = squeeze(Qk(j,:,:));
        
        xy = 1.0*plot_ellipse(inv(S_now));
        xyP = 1.0*plot_ellipse(inv(P_now));
        xyQ = 1.0*plot_ellipse(inv(Q_now));
    
        plot(phatStar(1,j) + xy(1,:),phatStar(2,j) + xy(2,:),'k');
        plot(phatStar(1,j) + xyP(1,:),phatStar(2,j) + xyP(2,:),'m');
        plot(phatStar(1,j) + xyQ(1,:),phatStar(2,j) + xyQ(2,:),'g');
        pp = sensor_state_log(1:2,j);
        thetap = sensor_state_log(3,j);
        xp = [cos(thetap) ; sin(thetap)];
        yp = [cos(thetap+pi/2) ; sin(thetap+pi/2)];
        
        plot([pp(1) pp(1)+0.5*xp(1)],[pp(2) pp(2)+0.5*xp(2)],'go-',MarkerSize=3,LineWidth=1)
        plot(pp(1),pp(2),'r+',MarkerSize=4,LineWidth=3)
    
    end
    axis equal
    
    
    figure()
    plot(t,vp_log)

    figure()
    plot(t,VPk_log)

    
    
    input('Press enter to continue ... ')


end







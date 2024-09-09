function [dSdEta, dSde1, dSde2, dSde3] = calculate_dSigma(Sigma_c)

    syms eta
    syms e1
    syms e2
    syms e3
   
    
    R = quat2rot([eta;e1;e2;e3]);
    
    Sigma = R*Sigma_c*R';
    
    dSdEta = diff(Sigma,'eta');
    dSde1 = diff(Sigma,'e1');
    dSde2 = diff(Sigma,'e2');
    dSde3 = diff(Sigma,'e3');



end
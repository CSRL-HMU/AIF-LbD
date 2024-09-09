function [dRdEta, dRde1, dRde2, dRde3] = calculate_dR()

    syms eta
    syms e1
    syms e2
    syms e3 
    
    R = quat2rot([eta;e1;e2;e3]);
    
    dRdEta = diff(R,'eta');
    dRde1 = diff(R,'e1');
    dRde2 = diff(R,'e2');
    dRde3 = diff(R,'e3');

end
% Skew Symmetric Matrix
function S = skewSymmetric(x)
    S = [0, -x(3), x(2); x(3), 0, -x(1); -x(2), x(1), 0];
end

% Inverse Skew Symmetric Matrix
function x = skewSymmetricInv(S)
    x = [(S(3,2) - S(2,3)) / 2; (S(1,3) - S(3,1)) / 2; (S(2,1) - S(1,2)) / 2];
end

% 5th Order Polynomial
function retTemp = get5thOrder(t, p0, pT, totalTime)
    p0 = p0(:); % Ensure column vector
    pT = pT(:);

    retTemp = zeros(length(p0), 3);
    
    if t < 0
        retTemp(:,1) = p0;
    elseif t > totalTime
        retTemp(:,1) = pT;
    else
        tau = t / totalTime;
        retTemp(:,1) = p0 + (pT - p0) * (10 * tau^3 - 15 * tau^4 + 6 * tau^5);
        retTemp(:,2) = (pT - p0) * (30 * t^2 / totalTime^3 - 60 * t^3 / totalTime^4 + 30 * t^4 / totalTime^5);
        retTemp(:,3) = (pT - p0) * (60 * t / totalTime^3 - 180 * t^2 / totalTime^4 + 120 * t^3 / totalTime^5);
    end
end
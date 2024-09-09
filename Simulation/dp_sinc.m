
function y = dp_sinc(x)
    if x == 0
        y = 1.0;
    else
        y = sin(x) / x;
    end
end
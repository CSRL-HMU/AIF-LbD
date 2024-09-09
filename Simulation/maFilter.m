% Moving Average Filter
function y = maFilter(x, ncoeffs)
    MA_coeffs = ones(1, ncoeffs) / ncoeffs;
    y_f = conv(x, MA_coeffs);
    y = y_f(end - length(x) + 1 : end);
end

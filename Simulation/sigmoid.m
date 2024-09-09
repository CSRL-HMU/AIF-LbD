% Sigmoid function
function y = sigmoid(x, c, h)
    y = get5thOrder(x - (c - 0.5 * h), 0, 1, h);
    y = y(1,1);
end
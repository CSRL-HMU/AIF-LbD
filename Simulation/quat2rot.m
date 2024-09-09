function R = quat2rot(Q)

    n = Q(1);
    ex = Q(2);
    ey = Q(3);
    ez = Q(4);
    
   

    R(1, 1) = 2 * (n * n + ex * ex) - 1;
    R(1, 2) = 2 * (ex * ey - n * ez);
    R(1, 3) = 2 * (ex * ez + n * ey);

    R(2, 1) = 2 * (ex * ey + n * ez);
    R(2, 2) = 2 * (n * n + ey * ey) - 1;
    R(2, 3) = 2 * (ey * ez - n * ex);

    R(3, 1) = 2 * (ex * ez - n * ey);
    R(3, 2) = 2 * (ey * ez + n * ex);
    R(3, 3) = 2 * (n * n + ez * ez) - 1;

end
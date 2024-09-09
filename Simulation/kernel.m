
classdef kernel
    properties
        type
        h
        c
    end
    methods
        % Constructor
        function obj = kernel(type_in, h_in, c_in)
            obj.type = type_in;
            obj.h = h_in;
            obj.c = c_in;
        end
        
        % Kernel evaluation method
        function val = psi(obj, x)
            if strcmp(obj.type, 'sinc')
                val = dp_sinc(obj.h * pi * (x - obj.c));
            else
                val = exp(-obj.h * (x - obj.c)^2);
            end
        end

           % works only for gaussian kernels
        function val = dpsi(obj, x)
          
            val = -2*obj.h * (x - obj.c)* exp(-obj.h * (x - obj.c)^2);
            
        end
        
        % Plotting the kernel
        function plot(obj)
            N_points = 1000;
            x_array = linspace(obj.c - 4 * obj.h, obj.c + 4 * obj.h, N_points);
            y_array = zeros(1, N_points);
            
            for i = 1:N_points
                y_array(i) = obj.psi(x_array(i));
            end
            
            plot(x_array, y_array);
            xlabel('$x$', 'FontSize', 14, Interpreter='latex');
            ylabel('$\psi(x)$', 'FontSize', 14, Interpreter='latex');
            title(sprintf('%s kernel function. c=%f, h=%f', obj.type, obj.c, obj.h), 'FontSize', 14);
            grid on;
        end
    end
end

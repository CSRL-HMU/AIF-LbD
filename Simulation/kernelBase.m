
classdef kernelBase
    properties
        N
        T
        kernelType
        xType
        a_x
        c_t
        h_t
        kernelArray
    end
    methods
        % Constructor
        function obj = kernelBase(N_in, T_in, kernelType_in, xType_in, a_x_in)
            obj.N = N_in;
            obj.T = T_in;
            obj.kernelType = kernelType_in;
            obj.xType = xType_in;
            obj.a_x = a_x_in;
            obj.c_t = linspace(0, obj.T, obj.N);
            
            if strcmp(obj.kernelType, 'sinc')
                obj.h_t = 1 / (obj.c_t(3) - obj.c_t(2));
            else
                obj.h_t = 1 / (obj.c_t(3) - obj.c_t(2))^2;
            end
            
            obj.kernelArray = cell(1, obj.N);
            for i = 1:obj.N
                obj.kernelArray{i} = kernel(obj.kernelType, obj.h_t, obj.c_t(i));
            end
        end
        
        % Mapping time -> phase variable
        function x = ksi(obj, t_in)
            if strcmp(obj.xType, 'linear')
                if t_in < obj.T
                    x = t_in / obj.T;
                else
                    x = 1;
                end
            else
                x = exp(-obj.a_x * t_in);
            end
        end
        
        % Inverse mapping phase variable -> time
        function t = ksi_inv(obj, x_in)
            if strcmp(obj.xType, 'linear')
                t = x_in * obj.T;
            else
                t = -log(x_in) / obj.a_x;
            end
        end
        
        % Get psi values at a given phase variable x
        function psi_vals = get_psi_vals_x(obj, x_in)
            t = obj.ksi_inv(x_in);
            psi_vals = obj.get_psi_vals_t(t);
        end

        %WORKS ONLY FOR LINEAR CANONICAL
        function dpsi_vals = get_dpsi_vals_x(obj, x_in)
            t = obj.ksi_inv(x_in);
            dpsi_vals = obj.get_dpsi_vals_t(t)*obj.T;
        end

      
        
        % Get psi values at a given time t
        function psi_vals = get_psi_vals_t(obj, t_in)
            psi_vals = zeros(1, obj.N);
            for i = 1:obj.N
                psi_vals(i) = obj.kernelArray{i}.psi(t_in);
            end
        end
        
        %WORKS ONLY FOR LINEAR CANONICAL
        function dpsi_vals = get_dpsi_vals_t(obj, t_in)
            dpsi_vals = zeros(1, obj.N);
            for i = 1:obj.N
                dpsi_vals(i) = obj.kernelArray{i}.dpsi(t_in);
            end
        end


        % Plot the kernel base in time
        function plot_t(obj)
            N_points = 1000;
            t_array = linspace(0, 1.2 * obj.T, N_points);
            y_array = zeros(obj.N, N_points);
            
            for i = 1:N_points
                y_array(:, i) = obj.get_psi_vals_t(t_array(i));
            end
            
            for i = 1:obj.N
                plot(t_array, y_array(i, :));
                hold on;
            end
            xlabel('$t$ (s)', 'FontSize', 14, Interpreter='latex');
            ylabel('$\psi_i(t)$', 'FontSize', 14, Interpreter='latex');
            title(sprintf('Kernel bases in t. N=%d', obj.N), 'FontSize', 14, Interpreter='latex');
            grid on;
            hold off;
        end
        
        % Plot the kernel base as a function of phase variable x
        function plot_x(obj)
            N_points = 1000;
            
            if strcmp(obj.xType, 'linear')
                xmin = 0;
            else
                xmin = obj.ksi(obj.T);
            end
            
            x_array = linspace(xmin, 1, N_points);
            y_array = zeros(obj.N, N_points);
            
            for i = 1:N_points
                y_array(:, i) = obj.get_psi_vals_x(x_array(i));
            end
            
            for i = 1:obj.N
                plot(x_array, y_array(i, :));
                hold on;
            end
            xlabel('$x$', 'FontSize', 14, Interpreter='latex');
            ylabel('$\psi_i$(x)', 'FontSize', 14, Interpreter='latex');
            title(sprintf('Kernel bases in x. N=%d', obj.N), 'FontSize', 14, Interpreter='latex');
            grid on;
            hold off;
        end
    end
end

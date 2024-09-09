
classdef dmp
    properties
        N
        T
        kernelType
        xType
        a_x
        a
        b
        W
        x0
        tau
        y0
        g
        kb
    end
    methods
        % Constructor
        function obj = dmp(N_in, T_in, kernelType_in, xType_in, a_x_in, a_in, b_in, tau_in)
            if nargin < 7
                a_in = 20.0;
                b_in = 0.8;
                tau_in = 1.0;
            end
            obj.N = N_in;
            obj.T = T_in;
            obj.kernelType = kernelType_in;
            obj.xType = xType_in;
            obj.a_x = a_x_in;
            obj.a = a_in;
            obj.b = b_in;
            obj.W = zeros(N_in, 1);
            obj.tau = tau_in;
            obj.kb = kernelBase(obj.N, obj.T, obj.kernelType, obj.xType, obj.a_x);

            if strcmp(xType_in, 'linear')
                obj.x0 = 0.0;
            else
                obj.x0 = 1.0;
            end
        end
        
        % Set initial position
        function obj = set_init_position(obj, y0_in)
            obj.y0 = y0_in;
        end
        
        % Set goal
        function obj = set_goal(obj, g_in)
            obj.g = g_in;
        end
        
        % Set time scaling parameter
        function obj = set_tau(obj, tau_in)
            obj.tau = tau_in;
        end
        
        % Get state derivative
        function [xdot, ydot, zdot] = get_state_dot(obj, x, y, z, customScale, scalingTerm)
            if nargin < 5
                customScale = false;
                scalingTerm = 1;
            end
            
            if strcmp(obj.xType, 'linear')
                xdot = 1.0 / obj.T;
            else
                xdot = -obj.a_x * x;
            end
            
            ydot = z;
            Psi = obj.kb.get_psi_vals_x(x);

        
            
            if strcmp(obj.kernelType, 'sinc')
                f = obj.W' * Psi';
            else
                f = obj.W' * Psi' / sum(Psi);
            end
            
            s = 1 - sigmoid(obj.kb.ksi_inv(x), 1.0 * obj.T, 0.05 * obj.T);
            if ~customScale
                scalingTerm = (obj.g - obj.y0);
            end


            zdot = obj.a * (obj.b * (obj.g - y) - z) + s * scalingTerm * f;
            
            xdot = xdot / obj.tau;
            ydot = ydot / obj.tau;
         
            zdot = zdot / obj.tau;
        end
        
        % Training method
        function obj = train(obj, dt, y_array, plotEn, customScale, scalingTerm)
            if nargin < 4
                plotEn = false;
                customScale = false;
                scalingTerm = 1;
            end
            
            y_array = y_array(:);
        
            Npoints = length(y_array);
            t_array = (0:Npoints - 1) * dt;
            obj.T = t_array(end);
            obj.tau = 1.0;
            obj.y0 = y_array(1);
            obj.g = y_array(end);

            
            obj.kb = kernelBase(obj.N, obj.T, obj.kernelType, obj.xType, obj.a_x);
            z_array = [0;diff(y_array)]/dt;
      
            zdot_array = [0;diff(y_array)]/dt;
            x_array = zeros(1, Npoints);
            
            
            for i = 1:Npoints
                x_array(i) = obj.kb.ksi(t_array(i));
            end

    
            
            if ~customScale
                scalingTerm = (obj.g - obj.y0);
            end
            
            fd_array = (zdot_array - obj.a * (obj.b * (obj.g - y_array) - z_array)) / scalingTerm;
            
            if strcmp(obj.kernelType, 'sinc')
                obj = obj.approximate_sincs(t_array, fd_array, Npoints, plotEn);
            else
                obj = obj.approximate_LS_gaussians(t_array, fd_array, Npoints, plotEn);
            end

            obj.kb
        end
        
        % Approximate sinc functions
        function obj = approximate_sincs(obj, t, fd, Npoints, plotEn)
            c = obj.kb.c_t;
            j = 1;
            
            for i = 1:Npoints
                if t(i) > c(j)
                    obj.W(j) = (fd(i) + fd(i - 1)) / 2.0;
                    j = j + 1;
                end
            end
            
            if plotEn
                PsiPsi = zeros(Npoints, obj.N);
                for i = 1:Npoints
                    Psi = obj.kb.get_psi_vals_t(t(i));
                    PsiPsi(i, :) = Psi;
                end
                
                f = PsiPsi * obj.W';
                plot(t, fd);
                hold on;
                plot(t, f);
                xlabel('t', 'FontSize', 14);
                ylabel('f(x(t))', 'FontSize', 14);
                title(sprintf('Function approximation with sinc base. N=%d', obj.N), 'FontSize', 14);
                grid on;
                hold off;
            end
        end
        
        % Approximate Gaussian functions
        function obj = approximate_LS_gaussians(obj, t, fd, Npoints, plotEn)
            PsiPsi = zeros(Npoints, obj.N);
            
            for i = 1:Npoints
                Psi = obj.kb.get_psi_vals_t(t(i));
                PsiPsi(i, :) = Psi / sum(Psi);
            end
            
            obj.W = pinv(PsiPsi) * fd;
            
            if plotEn
                f = PsiPsi * obj.W;
                plot(t, fd);
                hold on;
                plot(t, f);
                xlabel('t', 'FontSize', 14);
                ylabel('f(x(t))', 'FontSize', 14);
                title(sprintf('Function approximation with Gaussian base. N=%d', obj.N), 'FontSize', 14);
                grid on;
                hold off;
            end
        end

        % WORKS ONLY FOR GAUSSIAN KERNELS
        function df = getDf(obj, x_in)
            dpsi_array = obj.kb.get_dpsi_vals_x(x_in)';
            psi_array = obj.kb.get_psi_vals_x(x_in)';
            df = -obj.W'*(sum(dpsi_array)/(sum(psi_array)^2))*dpsi_array;
            

        end

    end
end

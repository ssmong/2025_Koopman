function vis_brs_options(Max_BRS, stats, n_z, w_lb, w_ub, n_samples, selected_options)
% VIS_BRS_OPTIONS
%   Visualizes BRS analysis results on the quaternion sphere using Mesh Interpolation.
%   Uses Fibonacci Lattice for uniform sampling on the visualization sphere.

    arguments
        Max_BRS
        stats
        n_z (1,1) double
        w_lb (3,1) double
        w_ub (3,1) double
        n_samples (1,1) double = 3000
        selected_options (1,:) double = [1, 2, 3]
    end

    idx_q = 1:4;
    idx_w = 5:7;

    if ~isfield(stats, 'x_mean') || ~isfield(stats, 'x_std')
        error('Statistics (x_mean, x_std) missing.');
    end
    x_mean = stats.x_mean;
    x_std  = stats.x_std;

    % q_mean, q_std are needed if we want to un-normalize or normalize samples
    % But samples are generated in physical space (Unit Sphere) then normalized.
    q_mean = x_mean(idx_q)'; q_std = x_std(idx_q)';
    w_mean = x_mean(idx_w)'; w_std = x_std(idx_w)';

    %% 2. Define Target Region (for Opt 3)
    Target_Vol_Ref = 0;
    Target_W_Box = [];
    
    run_opt3 = ismember(3, selected_options);
    
    if run_opt3
        Target_W_Box = Polyhedron('lb', w_lb, 'ub', w_ub);
        Target_Vol_Ref = Target_W_Box.volume();
    end

    %% 3. Generate Samples (Fibonacci Lattice on Unit Sphere)
    % These points represent (q1, q2, q3). We assume q0 = 0 (Pure Quaternion, 180 deg rotation)
    % or we treat them as just a visualization slice of the S3 hypersphere.
    
    points = fibonacci_sphere(n_samples);
    q234 = points'; % [3, N] -> q1, q2, q3
    q1 = zeros(1, n_samples); % q0 = 0
    
    q_phys = [q1; q234]; % [4, N] (Physical Space)
    
    % Normalize for BRS (Neural Network Input)
    q_norm = (q_phys - q_mean) ./ q_std;

    %% 4. Analyze BRS
    res_opt1_exist = false(n_samples, 1);
    res_opt2_rest  = false(n_samples, 1);
    res_opt3_vol   = zeros(n_samples, 1);

    z_nn_zero = zeros(n_z - 7, 1); 
    
    run_opt1 = ismember(1, selected_options);
    run_opt2 = ismember(2, selected_options);
    
    need_slicing = run_opt1 || run_opt3;

    fprintf('Running Analysis on %d samples (Options: %s)...\n', ...
            n_samples, num2str(selected_options));
    
    parfor i = 1:n_samples
        q_curr_norm = q_norm(:, i);
        
        % Option 1 & 3
        if need_slicing
            try
                Slice_P = Max_BRS.slice(idx_q, q_curr_norm);
                
                if ~isempty(z_nn_zero)
                    idx_others = setdiff(1:n_z, [idx_q, idx_w]);
                    Slice_P = Slice_P.slice(idx_others, z_nn_zero);
                end
                
                if ~Slice_P.isEmptySet()
                    if run_opt1, res_opt1_exist(i) = true; end
                    
                    if run_opt3
                        Safe_Region = Slice_P.intersect(Target_W_Box);
                        if ~Safe_Region.isEmptySet()
                            res_opt3_vol(i) = Safe_Region.volume() / Target_Vol_Ref;
                        end
                    end
                else
                    % Empty
                end
            catch
            end
        end
        
        % Option 2
        if run_opt2
            % Rest (w=0) in Normalized Space
            % w=0 physical -> (0 - mean) / std
            w_rest_norm = (zeros(3,1) - w_mean) ./ w_std;
            z_test = [q_curr_norm; w_rest_norm; z_nn_zero];
            if Max_BRS.contains(z_test)
                res_opt2_rest(i) = true;
            end
        end
    end
    fprintf('Analysis Done.\n');

    %% 5. Visualization (Mesh)
    
    % Option 1: Existence
    if run_opt1
        plot_mesh_on_sphere(q_phys, double(res_opt1_exist), ...
            'Option 1: Existence (Mesh)', 'Existence (0=Unsafe, 1=Safe)');
    end

    % Option 2: Rest-to-Rest
    if run_opt2
        plot_mesh_on_sphere(q_phys, double(res_opt2_rest), ...
            'Option 2: Rest-to-Rest (Mesh)', 'Rest-to-Rest Capable');
    end

    % Option 3: Safety Margin
    if run_opt3
        plot_mesh_on_sphere(q_phys, res_opt3_vol, ...
            'Option 3: Safety Margin (Mesh)', 'Safety Margin (Vol Ratio)');
    end

end

function points = fibonacci_sphere(samples)
    % Generates points on a unit sphere using Fibonacci Lattice
    points = zeros(samples, 3);
    phi = pi * (3 - sqrt(5)); % Golden angle
    
    for i = 0:samples-1
        y = 1 - (i / (samples - 1)) * 2; % y goes from 1 to -1
        radius = sqrt(1 - y * y); % radius at y
        theta = phi * i; % golden angle increment
        
        x = cos(theta) * radius;
        z = sin(theta) * radius;
        
        points(i+1, :) = [x, y, z];
    end
end

function plot_mesh_on_sphere(q_phys, values, fig_name, cbar_label)
    % PLOT_MESH_ON_SPHERE
    %   Interpolates values on the quaternion sphere and plots a mesh.
    
    x = q_phys(2, :)';
    y = q_phys(3, :)';
    z = q_phys(4, :)';
    
    try
        dt = delaunayTriangulation(x, y, z);
        [tri, points] = freeBoundary(dt);
        
        % Interpolation
        F = scatteredInterpolant(x, y, z, values, 'linear', 'nearest');
        c_vals = F(points(:,1), points(:,2), points(:,3));
        
        figure('Name', fig_name, 'Color', 'w');
        
        trisurf(tri, points(:,1), points(:,2), points(:,3), ...
                'FaceVertexCData', c_vals, ...
                'FaceColor', 'interp', ...
                'EdgeColor', 'none', ...
                'FaceAlpha', 0.8);
        
        hold on; axis equal; grid on; view(3);
        camlight; lighting gouraud; material dull;
        
        c = colorbar;
        c.Label.String = cbar_label;
        clim([0 1]);
        
        colormap(jet); 
        
        title(fig_name);
        xlabel('q_1'); ylabel('q_2'); zlabel('q_3');
        
        % Draw Reference Unit Sphere Wireframe
        [sx, sy, sz] = sphere(30);
        surf(sx, sy, sz, 'FaceColor', 'none', 'EdgeColor', [0.8 0.8 0.8], 'FaceAlpha', 0.1);
        
    catch e
        warning('Mesh generation failed: %s. Not enough points or coplanar.', e.message);
    end
end
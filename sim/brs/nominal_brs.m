clear; clc; close all;

data_path = "mpbm_10_200_0.1.h5";
seq_id = 1;
checkpoint_dir = "2026-01-12/10-45-57";

setenv('SEQ_ID', char(string(seq_id)));
setenv('DATA_PATH', data_path);
setenv('CHECKPOINT_DIR', checkpoint_dir);

fprintf("Calling Python script to export matrices for Sequence %d from %s\n", seq_id, data_path);

currentDir = fileparts(mfilename('fullpath'));
pyScriptPath = fullfile(currentDir, 'export_lpv.py');
cmd = sprintf('python "%s"', pyScriptPath);

[status, result] = system(cmd);

if status ~= 0
    error('Failed to export matrices. MATLAB returned status %d:\n%s', status, result);
else
    fprintf("Python script completed successfully.\n");
end

projectRoot = fileparts(fileparts(currentDir));
matFilePath = fullfile(projectRoot, 'data', 'brs', 'brs_data.mat');

if ~isfile(matFilePath)
    error('MAT file was not found at expected path: %s', matFilePath);
else
    fprintf("Loading data from %s...\n", matFilePath);
    load(matFilePath);
end

sys = LTISystem('A', A, 'B', B);
n_z = size(A, 1);
n_u = size(B, 2);
sys.u.min = u_lb * ones(n_u, 1);
sys.u.max = u_ub * ones(n_u, 1);

% State Constraints (Angular Velocity Limit: +/- 0.5 rad/s)
sys.x.min = -Inf * ones(n_z, 1);
sys.x.max =  Inf * ones(n_z, 1);

% Assuming z structure preserves state at the beginning and w is at indices 5,6,7
idx_w = [5, 6, 7]; 
sys.x.min(idx_w) = w_lb;
sys.x.max(idx_w) = w_ub;

eps = 1e-4;

N = 20;


%% 1. Target Set Definition (Origin in Latent Space)
% Koopman Latent Space의 원점(Regulation 목표) 주변의 매우 작은 Box를 목표로 설정

Target = Polyhedron('lb', -eps * ones(n_z, 1), 'ub', eps * ones(n_z, 1));

%% 2. Maximal BRS Calculation (Loop)
% Maximal BRS = Union of all k-step backward reachable sets (k=0...N)
% Preallocate BRS_Array with Target (size N+1) to avoid dynamic resizing
BRS_Array(N+1) = Target; 
BRS_Array(1) = Target;
R_k = Target;

fprintf('Computing Maximal BRS for N = %d...\n', N);

for k = 1:N
    % 2.1 One-step Backward Reachable Set Calculation
    % P_prev = { z | exists u \in U s.t. A*z + B*u \in R_k }
    R_prev = sys.reachableSet('X', R_k, 'direction', 'backward', 'N', 1);
    
    % 2.2 Empty Set Check
    if R_prev.isEmptySet()
        warning('BRS became empty at step %d. Constraints might be too tight or system is unstable.', k);
        break;
    end
    
    % 2.3 Complexity Reduction (Crucial for High-dim)
    R_prev.minHRep();
    
    % 2.4 Update & Accumulate
    R_k = R_prev;
    BRS_Array(k+1) = R_k;
    
    fprintf('Step %d / %d completed. Number of inequalities: %d\n', k, N, size(R_k.H, 1));
end

%% 3. Create Maximal BRS Object
% PolyUnion을 사용하여 합집합 객체 생성
Max_BRS = PolyUnion(BRS_Array);

%% 4. Visualization & Verification
stats.x_mean = x_mean;
stats.x_std = x_std;

% Visualization Options
n_samples = 3000;
selected_opts = [1, 2, 3];

vis_brs_options(Max_BRS, stats, n_z, w_lb, w_ub, n_samples, selected_opts);

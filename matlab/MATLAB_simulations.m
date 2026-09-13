%% Canonical PSO vs FIPS 
% This code is based on the considerations made throughout the Section 5 of
% the related Thesis.

% This script compares three swarm configurations on three 2D benchmark
% functions (Sphere, Ackley, Rastrigin):
% 1) Canonical PSO    global topology, constriction factor
% 2) FIPS-Ring        unweighted FIPS, ring neighborhood topology
% 3) FIPS-All         unweighted FIPS, fully connected topology
% 
% For each benchmark function, RUNS = 20 independent runs are executed. 
% Within a given run, all three configurations start from the same 
% random initial positions/velocities (X0, V0), so that any performance 
% difference can be attributed to the algorithm/topology rather than
% to the initial conditions. The random seed is set once per run (based
% on the function index and run number, not on the algorithm), while the 
% random sequences used afterwards for r1, r2 and rK during the velocity 
% and position updates evolve indepepndently for each configuration.
% 
% For every run, the script records:
%     - the best fitness found at the end of the run (finalBest)
%     - whether/when the run satisfies the position-base success criterion, 
%     under both the standard threshold and a stricter one (successRun/ 
%     itersToHit and successRunTight/itersToHitThigh) 
%     - the mean convergence curve of the best fitness over iterations
% 
% Results are summarized in a table and in per-function and combined plots 
% on a logarithmic fitness scale.

% Author: Giorgia Colucci - Bachelor's Thesis, Politecnico di Torino

clear; clc; close all;
%% -------------------------- GLOBAL PARAMETERS --------------------------
D       = 2;            % search space dimensionality
N       = 30;           % swarm size
RUNS    = 20;           % independent runs per configuration
maxIter = 500;          % maximum number of iterations

% Canonical PSO parameters (constriction factor formulation, Eq. 3.4)
phi1 = 2.05;
phi2 = 2.05;
chi  = 0.7298;          % kappa = 1;

% FIPS parameters (unweighted, Eqs. 4.1, 4.3, 4.4)
phiFIPS = 4.1; 
% same chi as PSO

% Success criterion 
POS_TOL_FACTOR = 0.0015;
%% --------------------------- FUNCTIONS SETUP ---------------------------
% Standard 2D domains. All three functions have their known global optimum 
% at the origin, f(0,0) = 0.
funcNames = {'Sphere', 'Ackley', 'Rastrigin'};
lbAll    = [-100,       -32,       -5.12,];
ubAll    = [ 100,        32,        5.12,];

nFunc = length(funcNames);

% The three configurations compared for every function
confNames = {'Canonical PSO (global)', 'FIPS-Ring', 'FIPS-All'};
nConf     = length(confNames);

% Storage for the results (indices: function, configuration, run)
% 20-dimensional arrays, in which every element is a (nFunc x nConf)-dimensional matrix
finalBest   = zeros(nFunc, nConf, RUNS);    
successRun  = false(nFunc, nConf, RUNS);
itersToHit  = nan(nFunc, nConf, RUNS);      % NaN is used when a run is unsuccessful
successRunTight  = false(nFunc, nConf, RUNS);
itersToHitTight  = nan(nFunc, nConf, RUNS);

% Storage for the mean convergence curves (function, configuration, iter)
meanCurve   = zeros(nFunc, nConf, maxIter);

%% ------------------------ MAIN SIMULATIONS LOOP ------------------------
for fIdx = 1:nFunc
    funcActual = funcNames{fIdx};
    lb = lbAll(fIdx);
    ub = ubAll(fIdx);
    posTol = POS_TOL_FACTOR * (ub - lb);
    posTolTight = 1e-4 * (ub - lb);

    curveSum = zeros(nConf, maxIter);       % (3 x maxIter) - dimensional matrix

    for r = 1:RUNS
        
        % --- Reproducible seed, different for every run ---
        seed = 100*fIdx + r;   % seed is related to the benchmark function and to the run
        rng(seed);

        % Random initial conditions, shared by ALL THREE configurations
       
        X0 = lb + (ub - lb) * rand(N, D);      % Initialize random positions within bounds
        V0 = (rand(N,D) - 0.5) .* (ub - lb);   % Centering the random numbers to be drawn 
             % from the interval [-0.5, 0.5] and scaling the random velocity in relation 
             % to the search space

        % Configuration 1: Canonical PSO (global)
        % For each run, bc is a (1*maxIter) vector containing the best
        % fitness found at until every iteration.  Then, the values of every run
        % are additioned.
        [bc, bd] = runPSO(funcActual, lb, ub, N, D, maxIter, chi, phi1, phi2, X0, V0);
        finalBest(fIdx, 1, r) = bc(end);        % contains, for each run, the value of the best fitness reached 
                    % until the end of the simulation
        hitIdx = find(bd < posTol, 1, "first");     % find the index of the first elt that satisfies condition
        if ~isempty(hitIdx)     % following lines will be skipped if hitIdx is empy
            successRun(fIdx, 1, r) = true;      % true if criteria is reached
            itersToHit(fIdx, 1, r) = hitIdx;
        end
        curveSum(1, :) = curveSum(1,:) + bc;

        hitIdxTight = find(bd < posTolTight, 1, "first");  
        if ~isempty(hitIdxTight)    
            successRunTight(fIdx, 1, r) = true;     
            itersToHitTight(fIdx, 1, r) = hitIdxTight;
        end

        % Configuration 2: FIPS - Ring (same X0, V0)
        [bc, bd] = runFIPS(funcActual, lb, ub, N, D, maxIter, 'Ring', chi, phiFIPS, X0, V0);
        finalBest(fIdx, 2, r) = bc(end);
        hitIdx = find(bd < posTol, 1, "first");     % find the index of the first elt that satisfies condition
        if ~isempty(hitIdx)     % following lines will be skipped if hitIdx is empy
            successRun(fIdx, 2, r) = true;      % true if criteria is reached
            itersToHit(fIdx, 2, r) = hitIdx;
        end
        curveSum(2, :) = curveSum(2,:) + bc;

        hitIdxTight = find(bd < posTolTight, 1, "first");  
        if ~isempty(hitIdxTight)    
            successRunTight(fIdx, 2, r) = true;     
            itersToHitTight(fIdx, 2, r) = hitIdxTight;
        end

        % Configuration 3: FIPS - All (same X0, V0)
        [bc, bd] = runFIPS(funcActual, lb, ub, N, D, maxIter, 'All', chi, phiFIPS, X0, V0);
        finalBest(fIdx, 3, r) = bc(end);
        hitIdx = find(bd < posTol, 1, "first");     % find the index of the first elt that satisfies condition
        if ~isempty(hitIdx)     % following lines will be skipped if hitIdx is empy
            successRun(fIdx, 3, r) = true;      % true if criteria is reached
            itersToHit(fIdx, 3, r) = hitIdx;
        end
        curveSum(3, :) = curveSum(3,:) + bc;

        hitIdxTight = find(bd < posTolTight, 1, "first");  
        if ~isempty(hitIdxTight)    
            successRunTight(fIdx, 3, r) = true;     
            itersToHitTight(fIdx, 3, r) = hitIdxTight;
        end
    end

    meanCurve(fIdx, :, :) = curveSum / RUNS;        % mean convergence curve for each configuration: 
        % average of the minimum fitnesses found by the three configurations across all the runs 
    
end
%% ------------------------- PRINT SUMMARY TABLE -------------------------
fprintf('%-10s | %-22s | %-10s | %-11s | %-9s | %-7s | %-9s | %-8s | %-9s\n ', ...
    'Function', 'Configuration', 'Mean best', 'Median best', 'Std best', ...
    'Success', 'Med.iters', 'Succ-T', 'Med.iters-T');
fprintf(repmat('-', 1, 120)); fprintf('\n');

for fIdx = 1:nFunc
    for cIdx = 1:nConf
        % selects only the 20 run and eliminates the fIdx and cIdx
        % dimension with squeeze, for calculating the mean, median and std
        vals  = squeeze(finalBest(fIdx, cIdx, :));
        succ  = squeeze(successRun(fIdx, cIdx, :));
        iters = squeeze(itersToHit(fIdx, cIdx, :));
        succTight  = squeeze(successRunTight(fIdx, cIdx, :));
        itersTight = squeeze(itersToHitTight(fIdx, cIdx, :));

        medIters = median(iters(~isnan(iters)));    % eliminates all the nan values
        if isnan(medIters)
            medItersStr = 'never';       % if medIters is an empty arrays (every run had nan iters)
        else
            medItersStr = sprintf('%.1f', medIters);
        end

        medItersTight = median(itersTight(~isnan(itersTight)));    % eliminates all the nan values
        if isnan(medItersTight)
            medItersStrTight = 'never';       % if medIters is an empty arrays (every run had nan iters)
        else
            medItersStrTight = sprintf('%.1f', medItersTight);
        end

        fprintf('%-10s | %-22s | %-10.4g | %-11.4g | %-9.4g | %-7.2f | %-9s | %-8.2f | %-9s\n', ...
            funcNames{fIdx}, confNames{cIdx}, mean(vals), median(vals), std(vals), ...
            mean(succ)*100, medItersStr, mean(succTight)*100, medItersStrTight);
    end
end

%% -------------------------------- PLOTS --------------------------------
for fIdx = 1:nFunc

    figure('Name',['convergence_' funcNames{fIdx}]);
    hold on;

    plot(1:maxIter, squeeze(meanCurve(fIdx, 1, :)), 'k-', 'LineWidth', 2);
    plot(1:maxIter, squeeze(meanCurve(fIdx, 2, :)), 'b-', 'LineWidth', 1.5);
    plot(1:maxIter, squeeze(meanCurve(fIdx, 3, :)), 'r-', 'LineWidth', 1.5);

    % for dark plot
    % plot(1:maxIter, squeeze(meanCurve(fIdx, 1, :)), 'Color', '#00D7FF', 'LineWidth', 2, 'LineStyle', '-');
    % plot(1:maxIter, squeeze(meanCurve(fIdx, 2, :)), 'Color', '#FFAA00', 'LineWidth', 2, 'LineStyle', '--');
    % plot(1:maxIter, squeeze(meanCurve(fIdx, 3, :)), 'Color', '#FF55FF', 'LineWidth', 1.5);

    set(gca, 'Yscale', 'log');      % uses a logarithmic scale for the y-axes
    xlabel('Iteration');
    ylabel('Mean best fitness (log scale)');
    title(['Convergence on ' funcNames{fIdx} ' (mean over ' num2str(RUNS) ' runs']);
    legend(confNames);
    grid on;
    hold off;
end

%% --------------------------- COMBINED FIGURE ---------------------------
figCombined = figure('Name', 'convergence_all', 'Position', ...
    [100, 100, 1300, 1000]);

subplotIdx = {1, 2, [3, 4]};

if ~exist('figures', 'dir')
    mkdir('figures');
end

for fIdx = 1:nFunc
    subplot(2, 2, subplotIdx{fIdx});
    hold on;

    plot(1:maxIter, squeeze(meanCurve(fIdx, 1, :)), 'k-', 'LineWidth', 2);
    plot(1:maxIter, squeeze(meanCurve(fIdx, 2, :)), 'b-', 'LineWidth', 1.5);
    plot(1:maxIter, squeeze(meanCurve(fIdx, 3, :)), 'r-', 'LineWidth', 1.5);
    
    % for dark plot
    % plot(1:maxIter, squeeze(meanCurve(fIdx, 1, :)), 'Color', '#00D7FF', 'LineWidth', 2, 'LineStyle', '-');
    % plot(1:maxIter, squeeze(meanCurve(fIdx, 2, :)), 'Color', '#FFAA00', 'LineWidth', 2, 'LineStyle', '--');
    % plot(1:maxIter, squeeze(meanCurve(fIdx, 3, :)), 'Color', '#FF55FF', 'LineWidth', 2);

    set(gca, 'Yscale', 'log');      % uses a logarithmic scale for the y-axes
    xlabel('Iteration');
    ylabel('Mean best fitness (log scale)');
    title(['(' char(96 + fIdx) ')' funcNames{fIdx}]); 
    legend(confNames);
    grid on;
    set(gca, 'FontSize', 11);
    hold off;

    lgd = legend(confNames, 'Orientation', 'horizontal');
    lgd.Position(1) = 0.5 - lgd.Position(3)/2;      % horizontal centering
    lgd.Position(2) = 0.02;         % just above the bottom edge
end
    
exportgraphics(figCombined, fullfile('figures', 'convergence_all.png'), 'Resolution', 300);

%% --------------------------- PRESENTATION FIGURE ---------------------------
figCombined = figure('Name', 'convergence_all_presentation', ...
    'Position', [100, 100, 1600, 500]);

t = tiledlayout(1, 3, ...
    'TileSpacing', 'normal', ...
    'Padding', 'compact');

for fIdx = 1:nFunc

    ax = nexttile;
    hold on;

    % Canonical PSO
    p1 = plot(1:maxIter, squeeze(meanCurve(fIdx, 1, :)), ...
        'k-', 'LineWidth', 2);

    % FIPS-Ring
    p2 = plot(1:maxIter, squeeze(meanCurve(fIdx, 2, :)), ...
        'b-', 'LineWidth', 1.5);

    % FIPS-All
    p3 = plot(1:maxIter, squeeze(meanCurve(fIdx, 3, :)), ...
        'r-', 'LineWidth', 1.5);

    set(gca, 'YScale', 'log');

    xlabel('Iteration');
    ylabel('Mean best fitness');

    title(funcNames{fIdx});

    grid on;
    box on;

    set(gca, 'FontSize', 12);

    % Save handles only from the first subplot
    if fIdx == 1
        legendHandles = [p1 p2 p3];
    end

    hold off;
end

% One common legend for the entire figure
lgd = legend(legendHandles, confNames, ...
    'Orientation', 'horizontal');

lgd.Layout.Tile = 'south';

% Export
if ~exist('figures', 'dir')
    mkdir('figures');
end

exportgraphics(figCombined, ...
    fullfile('figures', 'convergence_all_presentation.png'), ...
    'Resolution', 300);
%% ========================================================================
% BENCHMARK FUNCITONS (2D formulations, Eqs. 5.1-5.3 of the thesis)
% =========================================================================
% Input:    function name funcActual and a N-by-2 matrix X (one particle 
% per row).
% Output:   N-by-1 vector representing the evaluations for every particle.
% X is always within [lb, ub] with the reflecting-wall strategy adopted.

function fval = evaluateFunc(funcActual, X)
    x1 = X(:, 1);
    x2 = X(:, 2);
    
    % when funcActual corresponds to the case_expression, the switch function
    % performs the corresponding statements and exits the swith lock.
    switch funcActual
        case 'Sphere'
            fval = x1.^2 + x2.^2;                                           % Sphere function evaluation
        case 'Ackley'
            a1 = sqrt(0.5 * (x1.^2 + x2.^2));
            a2 = 0.5 * (cos(2*pi*x1) + cos(2*pi*x2));
            fval = 20 + exp(1) - 20*exp(-0.2*a1) - exp(a2);                 % Ackley function evaluation
        case 'Rastrigin'
            fval = 20 + x1.^2 + x2.^2 - 10*cos(2*pi*x1) - 10*cos(2*pi*x2);  % Rastrigin function evaluation
        otherwise 
            error('Unknown benchmark function: %s', funcActual)
    end     
end

%% ========================================================================
% NEIGHBORHOOD TOPOLOGY (used by FIPS only)
% =========================================================================
% Returns the particle indices belonging to the neighborhood of particle i,
% including particle i itself. For Ring topology, neighborhood of particle 
% i has 3 members {i-1, i, i+1}. For All topology, has N members. 

function idxNeigh = getNeigh(topology, i , N) 
    if strcmp(topology, 'Ring')
        left = i - 1; 
            if left < 1; left = N; end
        right = i + 1;
            if right > N; right = 1; end
        idxNeigh = [left, i, right];
    else        % 'All'
        idxNeigh = 1:N;  % For 'All' topology, all particles are neighbors
    end
end

%% ========================================================================
% CANONICAL PSO, global cersion (constriction factor formulation, Eq. 3.4)
% =========================================================================
function [bestCurve, bestDist] = runPSO(funcActual, lb, ub, N, D, ...
    maxIter, chi, phi1, phi2, X, V)

    Pbest       = X;                                % initialize the BEST POSITION with X
    PbestVal    = evaluateFunc(funcActual, X);      % FITNESS of the best position

    bestCurve = zeros(1, maxIter);       % (1 x maxIter) - dimensional vector
    bestDist  = zeros(1, maxIter);       % distance from the known optimum

    for t = 1:maxIter

        % --- Global best used for velocity update ---
        [~, gBestIdx] = min(PbestVal);        % index of the best fitness value 
        Gbest = Pbest(gBestIdx, :);           % position (x1, x2) of the global best

        % --- Velocity and positions update (Eq. 3.4) ---
        r1 = rand(N, D);        % random values for every N particle and for the 2 dimensions
        r2 = rand(N, D);
        V = chi * (V + phi1 * r1 .* (Pbest - X) + phi2 * r2 .* (Gbest - X));
        X = X + V;

        % --- Reflecting wall boundary ---
        % The velocity component that caused the boundary violation is
        % reversed, and the position is clipped to the nearest bound
        hitLow  = X < lb;           % assign true only to the component outisde the bound
        hitHigh = X > ub;        
        V(hitLow | hitHigh) = -V(hitLow | hitHigh);   % change only component where the condition is true

        X = min(max(X, lb), ub);    % clip the position outside the boundary

        % --- Update personal bests AFTER the t velocity and position update ---
        fitVal = evaluateFunc(funcActual, X);
        for i = 1:N
            if fitVal(i) < PbestVal(i)
                Pbest(i, :) = X(i, :);      % Update best position of the single particle
                PbestVal(i) = fitVal(i);    % Update best fitness of the single particle
            end
        end

        % --- Global best position found so far AFTER iteration t ---
        % PbestVal contains time memory; while, bestVal is done among all particles
        [bestVal, gBestIdx] = min(PbestVal);        % minimum fitness and index of the best fitness 
                                                    % value among that of all the particles
        Gbest = Pbest(gBestIdx, :);                 % position (x1, x2) of the global best

        % --- Best curve and distance ---
        % Register, at every iteration, the quality of the solution found until now 
        bestCurve(t) = bestVal;             % minimum value of the best fitness value until iteration t
        bestDist(t)  = max(abs(Gbest));     % calculates max(|x_best|, |y_best|), i.e the distance
                                            % in INFINITY NORM. otherwise, can be used norm(Gbest)
    end 
end

%% ========================================================================
% FULLY INFORMED PARTICLE SWARM, unweighted (Eqs. 4.3 and 5.4)
% =========================================================================
function [bestCurve, bestDist] = runFIPS(funcActual, lb, ub, N, D, ...
    maxIter, topology, chi, phi, X, V)

    Pbest       = X;                                % initialize the BEST POSITION with X
    PbestVal    = evaluateFunc(funcActual, X);      % FITNESS of the best position

    bestCurve = zeros(1, maxIter);
    bestDist  = zeros(1, maxIter);       % distance from the known optimum

    for t = 1:maxIter
        Vnew = zeros(N, D);

        % --- Velocity and positions update (Eqs. 4.3 and 5.4) ---
        for i = 1:N
            idxNeigh = getNeigh(topology, i, N);
            K = length(idxNeigh);
            contribution = zeros(1,D);

            % Unweighted FIPS: average contribution of every neighbor
            for j = 1:K
                rj = rand(1, D);            % vector 1xD with random numbers from uniform distribution [0,1]
                phi_j = rj * (phi/K);   % scaling with (phi/K)
                contribution = contribution + phi_j .* (Pbest(idxNeigh(j), :) - X(i, :));
            end

            Vnew(i, :) = chi * (V(i, :) + contribution);
        end

        V = Vnew;
        X = X + V;

        % --- Reflecting wall boundary ---
        % The velocity component that caused the boundary violation is
        % reversed, and the position is clipped to the nearest bound
        hitLow  = X < lb;           % assign true only to the component outisde the bound
        hitHigh = X > ub;        
        
        V(hitLow | hitHigh) = -V(hitLow | hitHigh);   % change only component where the condition is true
        X = min(max(X, lb), ub);    % clip the position outside the boundary

        % --- Update personal bests AFTER the t velocity and position update ---
        fitVal = evaluateFunc(funcActual, X);
        for i = 1:N
            if fitVal(i) < PbestVal(i)
                Pbest(i, :) = X(i, :);      % Update best position of the single particle
                PbestVal(i) = fitVal(i);    % Update best fitness of the single particle
            end
        end

        % --- Global best position found so far AFTER iteration t ---
        [bestVal, bestIdx] = min(PbestVal);        % minimum value and index of the best fitness 
                                                   % value among that of all the particles
        bestPos = Pbest(bestIdx, :);               % position (x1, x2) of the global best

        % --- Best curve and distance ---
        % Register, at every iteration, the quality of the solution found until now 
        bestCurve(t) = bestVal;              % minimum value of the best fitness value
        bestDist(t)  = max(abs(bestPos));    % calculates max(|x_best|, |y_best|), i.e the distance
                                             % in infinity norm. otherwise, can be used norm(Gbest)
    end 
end
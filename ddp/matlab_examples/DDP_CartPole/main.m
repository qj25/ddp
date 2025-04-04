%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%          DDP Cart Pole           %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  Course: Robotics and Autonomy   %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  AE4803  Spring  2020            %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  Author: Alejandro Escontrela    %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

visualizing_bundles_exists = exist('visualizing_bundles', 'var');
visualizing_bundles = visualizing_bundles_exists && visualizing_bundles;

if ~visualizing_bundles
    close all;
    visualizing_bundles=false;
end

global Horizon;
global time;
global p_target
global Q_f;
global dt;
global mus;
global sigmas;


use_obstacles = true;
% Add obstacles to the cart x position. Obstacles add a "floor is lava"
% effect to the trajectory, whereby the robot will want to speed past
% obstacles to avoid accruing large losses.

if use_obstacles
    mus = [-0.5]; % Obstacle position.
    sigmas = [10]; % Obstacle weight.
else
    mus = [];
    sigmas = [];
end

% Environment parameters.
mc = 1.0; % Cart mass in Kg.
mp = 0.01; % Pole mass in Kg.
l = 0.25; % Length of the pole in m.
g = 9.8; % Gravity in m/s^2.

% Obtain expressions for F, Fx, Fu, & Fb.
dynamics = fnDynamics(mc, mp, l, g);
dynamics;

% Solver parameters.
Horizon = 1000; % Time Horizon.
num_iter = 400; % Number of Iterations
dt = 0.01; % Discretization.

% Costs.
Q_f = zeros(4,4); % State cost. 4x4 since state is 4-dimensional.
Q_f(1,1) = 5000; % X position cost.
Q_f(2,2) = 100;  % X velocity cost.
Q_f(3,3) = 15000; % Pole angle cost.
Q_f(4,4) = 100; % Pole angular velocity cost.

if ~(all(eig(Q_f) >= 0))
    error('Cost matrix Q_f not positive semi-definite.')
end

R = 15 * eye(1,1); % Control cost. 1x1 since control is 1-dimensional.

% Initialize solution.
% State represented as [x, x_dot, theta, theta_dot].
xo = zeros(4,1);
x_dim = length(xo);
x_traj = zeros(x_dim,Horizon); % Initial trajectory.

u_k = zeros(1,Horizon-1); % Initial control.
u_dim = size(u_k, 1);
du_k = zeros(u_dim,Horizon-1); % Initial control variation.

Cost = zeros(1, num_iter); % Cost history.
residuals = zeros(1, num_iter); % Residual history.

% Goal state:
p_target = zeros(x_dim, 1);
p_target(1,1) = -1.0;
p_target(2,1) = 0.0; % Target x_dot.
p_target(3,1) = pi; % Target theta.
p_target(4,1) = 0.0; % Target theta_dot.

% Add noise.
sigma = 0.0;

% Learning Rate
gamma = 0.2;

for k = 1:num_iter
    % Preallocate cost memory.
    q0 = zeros(Horizon-1);
    q_k = zeros(x_dim, Horizon-1);
    Q_k = zeros(x_dim, x_dim, Horizon-1);
    r_k = zeros(u_dim, Horizon-1);
    R_k = zeros(u_dim, u_dim, Horizon-1);
    P_k = zeros(u_dim, x_dim, Horizon-1);
    A = zeros(x_dim, x_dim, Horizon-1);
    B = zeros(x_dim, u_dim, Horizon-1);
    for  j = 1:(Horizon-1)    
        [l0,l_x,l_xx,l_u,l_uu,l_ux] = fnCost(x_traj(:,j), u_k(:,j), j,R,dt);
        % Compute loss function gradients for the current timestep.
        % Quadratic Approximations of the cost function.
        q0(j) = dt * l0; % L.
        q_k(:,j) = dt * l_x; % Lx.
        Q_k(:,:,j) = dt * l_xx; % Lxx.
        r_k(:,j) = dt * l_u; % Lu.
        R_k(:,:,j) = dt * l_uu; % Luu.
        P_k(:,:,j) = dt * l_ux; % Lux.

        % Linearize the dynamics using first order taylor expansion.
        [fx,fu] = fnState_And_Control_Transition_Matrices(x_traj(:,j),u_k(:,j),du_k(:,j),dt, dynamics);
        A(:,:,j) = eye(x_dim,x_dim) + fx * dt;
        B(:,:,j) = fu * dt;  
    end

    % Preallocate value function memory.
    Vxx = zeros(x_dim,x_dim,Horizon);
    Vx = zeros(x_dim, Horizon);
    V = zeros(1, Horizon);
 
    % Compute value function at final timestep, its gradient, and its jacobian.
    Vxx(:,:,Horizon)= Q_f;
    Vx(:,Horizon) = Q_f * (x_traj(:,Horizon) - p_target); 
    V(Horizon) = 0.5 * (x_traj(:,Horizon) - p_target)' * Q_f * (x_traj(:,Horizon) - p_target);
    
    % Backpropagation of the Value Function.
    for j = (Horizon-1):-1:1
        % Quu = Luu + B^T * Vxx * B
        H = R_k(:,:,j) + B(:,:,j)' * Vxx(:,:,j+1) * B(:,:,j);
        % Qux = Lux + B^T * Vxx * A
        G = P_k(:,:,j) + B(:,:,j)' * Vxx(:,:,j+1) * A(:,:,j);
        % Qu = Lu + B^T * Vx
        g_ = r_k(:,j) +  B(:,:,j)' * Vx(:,j+1);


        inv_H = inv(H); % Quu^-1
        L_k(:,:,j)= - inv_H * G; % Feedback term = -Quu^-1 * Qux.
        l_k (:,j) = - inv_H *g_; % Feedforward term = -Quu^-1 * Qu.

        % Vxx = (Lxx + A^T * Vxx * A) + (-Qxu * Quu^-1) * Quu * (-Quu^-1 * Qux)
        % + (-Qxu * Quu^-1 * Qux) + (Qxu * -Quu^-1 * Qux) 
        Vxx(:,:,j) = Q_k(:,:,j)+ A(:,:,j)' * Vxx(:,:,j+1) * A(:,:,j) + L_k(:,:,j)' * H * L_k(:,:,j) + L_k(:,:,j)' * G + G' * L_k(:,:,j);
        % Vx = (Lx + A^T * Vx') + (-Qxu * Quu^-1 * Qu) + (Qxu * -Quu^-1 * Qu) +
        % (-Qxu * Quu^-1 * Qu * Quu * -Quu^-1 * Qu)
        Vx(:,j)= q_k(:,j) +  A(:,:,j)' *  Vx(:,j+1) + L_k(:,:,j)' * g_ + G' * l_k(:,j) + L_k(:,:,j)'*H * l_k(:,j);
        % V = L + V' + (0.5 * Qu^T * Quu^-1 * Quu * Quu^-1 * Qu) + (-Qu^T * Quu^-1 * Qu)
        V(:,j) = q0(j) + V(j+1) + 0.5 *  l_k (:,j)' * H * l_k (:,j) + l_k (:,j)' * g_;
    end

    % Preallocate control variation memory.
    u_new = zeros(u_dim, Horizon-1);
    dx = zeros(x_dim,1);
    for i=1:(Horizon-1)
        % Find the controls.
        du = l_k(:,i) + L_k(:,:,i) * dx;
        dx = A(:,:,i) * dx + B(:,:,i) * du;  
        u_new(:,i) = u_k(:,i) + gamma * du;
    end

    u_k = u_new;

    % Create new rollout.
    [x_traj] = fnSimulate(xo,u_new,Horizon,dt,sigma, dynamics);
    Cost(:,k) = fnCostComputation(x_traj,u_k,p_target,dt,Q_f,R);
    if (k ~= 1)
        residuals(:, k) = abs(Cost(:, k - 1) - Cost(:, k));
    end
%     x1(k,:) = x_traj(1,:);

    if mod(k, 10) == 0
        fprintf('iLQG Iteration %d,  Current Cost = %e \n',k,Cost(1,k));
    end

end

residuals(:,1) = residuals(:,2);

time = zeros(1,Horizon);
time(1)=0;
for i= 2:Horizon
    time(i) =time(i-1) + dt;  
end

if ~visualizing_bundles
    visualize;
%     close(fh);
end

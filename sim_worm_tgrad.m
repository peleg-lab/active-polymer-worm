function [forcess,positions,forcess_init,positions_init,x] = sim_worm_tgrad(N,N_chains,T,Thigh,eLJ_inter,eLJ_intra,spring_coeff,bend_coeff,active_coeff,fvar,sigma,steps)
% Aggregate worms from random initial position then translate so CoM is at
% (0,0)
% Model worm movement as active Brownian polymer subject to Lennard-Jones, spring,
% and bending potentials and active tangential force
% Temperature gradient in space
% Inputs:
% N: number of monomers per worm
% N_chains: number of worms
% T: temperature
% eLJ_inter: L-J force factor for between-worm forces
% eLJ_intra: L-J force factor for within-worm forces (smaller than
% eLJ_inter)
% spring_coeff: spring coefficient
% bend_coeff: bending coefficient
% active_coeff: active force
% fvar: variance in the active force (scalar)
% sigma: equilibrium distance (default = 1.189 from modified L-J force)
% steps: number of time steps in simulation (default = 100000)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% June 2021
% Authors:
% Chantal Nguyen, chantal.nguyen@colorado.edu
% Orit Peleg, orit.peleg@colorado.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if nargin < 10
    fvar = 0; % variance in active force
    sigma = 2^(1/4);
    steps = 100000;
elseif nargin < 11
    sigma =  2^(1/4);
    steps = 100000;
elseif nargin < 12
    steps = 100000;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize parameters
dt=0.00001; % integration time step
L = (sigma/2)*sigma*sqrt(N_chains)*sqrt(N); % characteristic system size for visualization
print_interval = 1000; % how often to print the system's new conformation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize position velocities and forces
% first initialize to random position and run 120000 steps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize position velocities and forces
init_steps = 120000;
forcess_init = zeros(N,2,N_chains,init_steps/print_interval);
positions_init = zeros(N,2,N_chains,init_steps/print_interval);
x = NaN*positions_init(:,:,:,1);
init_active_coeff = 30.*ones(N_chains,N) + fvar*(rand(N_chains,N) - 0.5);
n_pre_steps = 10000;
step_grad = linspace(dt/10000000,dt,n_pre_steps);
while isnan(sum(sum(sum(x))))
    x = initial_configuration(sigma,N,N_chains);
    for pre_step = 1:n_pre_steps
        [x,~] = steepest_descent (N,N_chains,x,step_grad(pre_step),eLJ_inter,eLJ_intra,sigma,spring_coeff,bend_coeff,init_active_coeff,T); % calculate new positions based on forces
    end
end
% main dynamics loop
for step_i=1:init_steps
    [x,~,F] = steepest_descent (N,N_chains,x,dt,eLJ_inter,eLJ_intra,sigma,spring_coeff,bend_coeff,init_active_coeff,T); % calculate new positions based on forces
    
    if mod(step_i-1,print_interval)==0
        positions_init(:,:,:,(step_i-1)/print_interval+1) = x;
        forcess_init(:,:,:,(step_i-1)/print_interval+1) = F;
    end
end
% then subtract CoM location
x2 = reshape(permute(x,[1,3,2]),N*N_chains,2);
com = nansum(x2)/(N*N_chains);
com = repmat(com,N,1);
com = repmat(com,1,1,N_chains);
x = x - com; % shift x so center of mass is at 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
fa = zeros(N_chains,N);
fa(:,(N-round(.1*N)):end) = active_coeff;
active_coeff = fa;

forcess = zeros(N,2,N_chains,steps/print_interval);
positions = zeros(N,2,N_chains,steps/print_interval);

% temperature gradient
Tlow = 0;
Tgrad = linspace(Tlow,Thigh,100)';
Tgrad = repmat(Tgrad,1,100);
Tmap = @(x) mmap(x,-L*3,L*3,Tlow,Thigh);

% main dynamics loop

for step_i=1:steps
    
    
    [x,~,F] = steepest_descent (N,N_chains,x,dt,eLJ_inter,eLJ_intra,sigma,spring_coeff,bend_coeff,active_coeff,Tmap); % calculate new positions based on forces
    if mod(step_i-1,print_interval)==0
        positions(:,:,:,(step_i-1)/print_interval+1) = x;
        forcess(:,:,:,(step_i-1)/print_interval+1) = F;
    end
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x= initial_configuration (initial_sigma, N, N_chains)

% each worm is randomly positioned and oriented

x = zeros(N,2,N_chains);
R = (initial_sigma*N - initial_sigma*N/2)-(initial_sigma - initial_sigma*N/2);
angles = 2*pi*rand(N_chains,1);
for i = 1:N_chains
    temp = zeros(N,2);
    angle = angles(i);
    startx = rand*.5*initial_sigma*N - .25*initial_sigma*N;
    starty = rand*.5*initial_sigma*N - .25*initial_sigma*N;
    lastx = startx+R*cos(angle);
    lasty = starty+R*sin(angle);
    temp(:,1) = linspace(lastx,startx,N);
    temp(:,2) = linspace(lasty,starty,N);
    x(:,:,i) = temp;
    
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x,pairs,F,F_inter,F_intra,F_spring,F_act,F_bend] = steepest_descent (N, N_chains, x, dt, epsilon_LJ, eLJ_intra, sigma, spring_coeff, bend_coeff,active_coeff,T)
[F_particles,~,pairs,F_inter,F_intra,F_spring,F_act,F_bend] = forces(N,N_chains,x,epsilon_LJ, eLJ_intra, sigma,spring_coeff,bend_coeff,active_coeff);
F = F_particles;
if isnumeric(T)
    x = x + (dt*F) + T.*randn(size(x)); % Brownian fluctuation
else
    Tvals = x;
    Tvals(:,1,:) = Tvals(:,2,:);
    Tvals = T(Tvals);
    x = x + (dt*F)+ Tvals.*randn(size(x));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ip,pair,connector]= LJ_interactions (N,N_chains,x, sigma) % obtain inter-worm LJ interacting pairs
ip=0; connector=zeros((N*N_chains)^2,2); pair=zeros((N*N_chains)^2,4);
for chain1 = 1:N_chains
    for i=1:N
        for chain2 = (chain1+1):N_chains
            for j=1:N
                distance = (x(j,:,chain2)-x(i,:,chain1));
                if distance < sigma*3
                    ip = ip + 1; % interaction pair counter
                    pair(ip,:) = [i j chain1 chain2]; % particle numbers (i,j) belonging to pair (ip)
                    connector(ip,:) = distance;
                end
            end
        end
    end
end

connector = connector(1:ip,:);
pair = pair(1:ip,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ip,pair,connector]= intra_worm_LJ (N,N_chains,x, sigma) % obtain intra-worm LJ interacting pairs
ip=0; connector=zeros(N^2*N_chains,2); pair=zeros(N^2*N_chains,4);
for chain1 = 1:N_chains
    for i=1:N-1
        for chain2 = (chain1)
            for j=i+1:N
                distance = (x(j,:,chain2)-x(i,:,chain1));
                if norm(distance) < sigma*3
                    ip = ip + 1; % interaction pair counter
                    pair(ip,:) = [i j chain1 chain2]; % particle numbers (i,j) belonging to pair (ip)
                    connector(ip,:) = distance;
                end
            end
        end
    end
end
connector = connector(1:ip,:);
pair = pair(1:ip,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ip,pair,connector]= spring_interactions (N,N_chains,x) % obtain adjacent pairs
ip=0; connector=zeros(N*N_chains,2); pair=zeros(N*N_chains,3);
for chain = 1:N_chains
    for i=1:N-1
        j=i+1;
        distance = (x(j,:,chain)-x(i,:,chain));
        ip = ip + 1; %interaction pair counter
        pair(ip,:) = [i j chain]; % particle numbers (i,j) belonging to pair (ip)
        connector(ip,:) = distance;
    end
end

connector = connector(1:ip,:);
pair = pair(1:ip,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [F,P,pair,F_inter,F_intra,F_spring,F_act,F_bend] = forces (N,N_chains,x,eLJ_inter,eLJ_intra,sigma,spring_coeff,bend_coeff,active_coeff)
% calculate all forces
F=zeros(N,2,N_chains); P=zeros(N,2,N_chains);
F_inter=F;
F_intra=F;
F_spring=F;
F_act=F;
F_bend=F;
% inter-LJ forces
[no,pair,connector]=LJ_interactions(N,N_chains,x,sigma); %interacting pairs
for i=1:no
    FORCE=force_LJ(connector(i,:),eLJ_inter, sigma);
    F_inter(pair(i,1),:,pair(i,3))=F_inter(pair(i,1),:,pair(i,3))-FORCE;
    F_inter(pair(i,2),:,pair(i,4))=F_inter(pair(i,2),:,pair(i,4))+FORCE; % action = reaction
    P(pair(i,1),:,pair(i,3))=P(pair(i,1),:,pair(i,3))+(sum(FORCE.*connector(i,:)));
    P(pair(i,2),:,pair(i,4))=P(pair(i,2),:,pair(i,4))+(sum(FORCE.*connector(i,:)));
end
% intra-LJ forces
[no,pair,connector]=intra_worm_LJ(N,N_chains,x,sigma); %interacting pairs
for i=1:no
    FORCE=force_LJ(connector(i,:),eLJ_intra, sigma);
    F_intra(pair(i,1),:,pair(i,3))=F_intra(pair(i,1),:,pair(i,3))-FORCE;
    F_intra(pair(i,2),:,pair(i,4))=F_intra(pair(i,2),:,pair(i,4))+FORCE; % action = reaction
    P(pair(i,1),:,pair(i,3))=P(pair(i,1),:,pair(i,3))+(sum(FORCE.*connector(i,:)));
    P(pair(i,2),:,pair(i,4))=P(pair(i,2),:,pair(i,4))+(sum(FORCE.*connector(i,:)));
end
%spring forces:
[no,pair,connector]=spring_interactions(N,N_chains,x); %interacting pairs
for i=1:no
    FORCE = force_springs(connector(i,:),spring_coeff,sigma);
    F_spring(pair(i,1),:,pair(i,3))=F_spring(pair(i,1),:,pair(i,3))-FORCE;
    F_spring(pair(i,2),:,pair(i,3))=F_spring(pair(i,2),:,pair(i,3))+FORCE; % action = reaction
    P(pair(i,1),:,pair(i,3))=P(pair(i,1),:,pair(i,3))+(sum(FORCE.*connector(i,:)));
    P(pair(i,2),:,pair(i,3))=P(pair(i,2),:,pair(i,3))+(sum(FORCE.*connector(i,:)));
end
% active forces:
% active forces have constant magnitude active_coeff and act along the
% tangent of each bond, equally split between themonomers
for i = 1:no
    f_unit = connector(i,:)/norm(connector(i,:));
    FORCE = 0.5*(active_coeff(pair(i,3),pair(i,1))+active_coeff(pair(i,3),pair(i,2))).*f_unit;
    F_act(pair(i,1),:,pair(i,3)) = F_act(pair(i,1),:,pair(i,3))+FORCE/2;
    F_act(pair(i,2),:,pair(i,3)) = F_act(pair(i,2),:,pair(i,3))+FORCE/2;
end
% bending forces:
% bending forces calculated for every set of 3 adjacent particles
for chain = 1:N_chains
    for j = 2:(N-1)
        r1 = x(j-1,:,chain) - x(j,:,chain);
        r2 = x(j+1,:,chain) - x(j,:,chain);
        [force1,force2,force3] = force_bend(r1,r2,bend_coeff);
        F_bend(j-1,:,chain) = F_bend(j-1,:,chain) + force1;
        F_bend(j,:,chain) = F_bend(j,:,chain) + force2;
        F_bend(j+1,:,chain) = F_bend(j+1,:,chain) + force3;
    end
end
F =  F_inter + F_intra + F_act + F_spring + F_bend;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function curr_force = force_springs(r_vector,spring_coeff_array,sigma)
r2=sum(r_vector.^2,2);
r = sqrt(r2);
curr_force = zeros(length(r2), 2);
curr_force(:,1) = -spring_coeff_array'.*((r -  sigma)).*(r_vector(:,1)./r);
curr_force(:,2) = -spring_coeff_array'.*((r -  sigma)).*(r_vector(:,2)./r);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [force1,force2,force3] = force_bend(r_vector1,r_vector2,bend_coeff)
% equilibrium angle = pi
theta0 = pi;
% calculate angle formed by bonds on either side of central particle
theta = real(acos(complex(dot(r_vector1,r_vector2)./(norm(r_vector1)*norm(r_vector2)))));
if r_vector1(1) == -1*r_vector2(1) && r_vector1(2) == -1*r_vector2(2)
    theta = pi;
end
% bending force is -2 * kb * (theta - 0)
if (theta - theta0) ~= 0
    p1 = cross([r_vector1,0],cross([r_vector1,0],[r_vector2,0]));
    p1 = p1./norm(p1);
    p1 = p1(1:2);
    p3 = cross(-1*[r_vector2,0],cross([r_vector1,0],[r_vector2,0]));
    p3 = p3./norm(p3);
    p3 = p3(1:2);
    force1 = (-2*bend_coeff*(theta-theta0)/(norm(r_vector1)))*p1;
    force3 = (-2*bend_coeff*(theta-theta0)/(norm(r_vector2)))*p3;
    force2 = -1*force1 - force3;
else
    force1 = zeros(1,2);
    force2 = force1;
    force3 = force1;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function force_LJ= force_LJ (r_vector,epsilon_LJ, sigma)
r=norm(r_vector);
if r < sigma/2 % introduce cutoff at very small separations to avoid forces blowing up
    r = sigma/2;
    r_vector = (sigma/2)*r_vector./(norm(r_vector));
end
force_LJ = 16*epsilon_LJ*(2*r.^(-8)-r^(-4)) * r_vector; % modified Lennard Jones

end


function output = mmap(value,fromLow,fromHigh,toLow,toHigh)
narginchk(5,5)
nargoutchk(0,1)
output = (value - fromLow) .* (toHigh - toLow) ./ (fromHigh - fromLow) + toLow;
output(find(value < fromLow)) = toLow;
output(find(value > fromHigh)) = toHigh;

end

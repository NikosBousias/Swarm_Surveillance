clear all;
close all;
clc;

% mkdir video1
% video_name='RESILIENCE.avi'


%%%%%%%%%%%%%%%%%%% Set Simulation Options %%%%%%%%%%%%%%%%%%%
% Network options
% Altitude constraints
zmin_tot = 0.3;
zmax_tot = 2.3;

% Simulation options
% Simulation duration in seconds
Tfinal = 120;
% Time step in seconds
Tstep = 0.25;

% Control law options
% Planar control law gain
axy =0.5;
% Altitude control law gain
az = 0.25;
% Panning control law gain
ath = 0.0005;
% Tilting control law gain
ah = 0.0005;
% Zooming control law gain
azoom = 0.0005;

% Network plots to show during simulation
PLOT_STATE_2D = 0;
PLOT_STATE_3D = 1;
PLOT_STATE_QUALITY = 0;
SAVE_PLOTS = 0;

% Save simulation results to file
SAVE_RESULTS = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------ Set Parameters for sensing patterns ------------


% %range of delta
delta_max=[deg2rad(30)];
delta_min=[deg2rad(15)];

%limit tilt angle
h_lim=30;


% ---------------- Region ----------------

% %   Non-convex region
% Xb=[ 0 0 5 5 2.5 2.5 5 5];
% Yb=[ 0 5 5 3.5 3.5 1.5 1.5 0];
% [Xb, Yb] = poly2cw(Xb, Yb);
% region = [Xb ; Yb];
% region_area = polyarea( Xb, Yb );
% axis_scale = [-0.5 3 -0.5 3];

%   Non-convex region
Xb=[0 0 1.2 1.2 3 3 4 4 3.5 3.5 1 1];
Yb=[0 2 2 1.5 1.5 3 3 2 2 1 1 0];
[Xb, Yb] = poly2cw(Xb, Yb);
region = [Xb ; Yb];
region_area = polyarea( Xb, Yb );
axis_scale = [-0.5 3 -0.5 3];

% % Bullo region
% Xb=[ 0, 2.125, 2.9325, 2.975, 2.9325, 2.295, 0.85, 0.17 ];
% Yb=[ 0, 0, 1.5, 1.6, 1.7, 2.1, 2.3, 1.2 ];
% [Xb, Yb] = poly2cw(Xb, Yb);
% region = [Xb ; Yb];
% region_area = polyarea( Xb, Yb );
% axis_scale = [-0.5 3 -0.5 3];


%RANDOM CONVEX REGION
% xy = randn(10,2);
% CH = convhulln(xy)
%attempt to generate random points inside convex hull
% ntri=8;
% xy=region';
% xycent=mean(region',1);
% nxy = size(region',1);
% ncent = nxy+1;
% xy(ncent,:) = xycent;
% tri = [region',repmat(ncent,ntri,1)];
% % figure
% % plot(region(1,:),region(2,:),'bo');
% % hold on
% % plot([xy(tri(:,1),1),xy(tri(:,2),1),xy(tri(:,3),1)]',[xy(tri(:,1),2),xy(tri(:,2),2),xy(tri(:,3),2)]','g-')
% V = zeros(1,ntri);
% for ii = 1:ntri
%   V(ii) = abs(det(xy(tri(ii,1:2),:) - xycent));
% end
% V = V/sum(V)
% M = 1000;
% [~,~,simpind] = histcounts(rand(M,1),cumsum([0,V]));
% r1 = rand(M,1);
% uv = xy(tri(simpind,1),:).*r1 + xy(tri(simpind,2),:).*(1-r1);
% r2 = sqrt(rand(M,1));
% uv = uv.*r2 + xy(tri(simpind,3),:).*(1-r2);
% figure
% plot(xy(:,1),xy(:,2),'bo');
% hold on
% plot([xy(tri(:,1),1),xy(tri(:,2),1),xy(tri(:,3),1)]',[xy(tri(:,1),2),xy(tri(:,2),2),xy(tri(:,3),2)]','g-')
% plot(uv(:,1),uv(:,2),'m.')



% % Bullo region
% Xb=[ 2 4 -2 -4 ];
% Yb=[4 2 -4 -2 ];
% [Xb, Yb] = poly2cw(Xb, Yb);
% region = [Xb ; Yb];
% region_area = polyarea( Xb, Yb );
% axis_scale = [-0.5 3 -0.5 3];

% % Bullo region
% Xb=[ -5 -1 5 1];
% Yb=[-1 -5 1 5 ];
% [Xb, Yb] = poly2cw(Xb, Yb);
% region = [Xb ; Yb];
% region = 10 .* [Xb ; Yb];
% region_area = polyarea( Xb, Yb );
% axis_scale = [-0.5 3 -0.5 3];

% ---------------- Initial State ----------------
N=8;
X=[0.2808    0.4106    0.3214    0.0608    0.1919    0.0987    0.1471    0.1920];
Y=[0.8345    0.0993    1.8054    1.8896    0.9817    0.9785    0.6754    1.8001];
Z=[1.0385    0.5224    1.8605    1.0795    0.7834    1.1078    0.4929    0.5639];
%half the angle of conic field of view
delta=ones(1,N)*deg2rad(20);
% [X,Y]=random_initial(region,N);
% X = [1 2 0.5 2.2 0.8 1.7 1.3 1 1.3 1.6 1.8 1 2 1 2.2 0.8 1.7 1.3 1.3 1.3 1.6 1.8];
% Y = [1 1.8 1.5 0.3 0.5 2.1 1.3 2.2 0.8 1.6 1 1.8 1.5 0.3 0.5 2.1 1.5 0.5 0.8 1.6 0.8 1.6];
% Z = [1 1.1 0.5 0.8 1 2 1.6 0.4 0.9 1.3 1.5 1.8 1 1.1 0.5 0.8 1 2 1.6 0.4 0.9 1.3 1.5 1.8];
% TH = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
% HI=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
% Z=zmin_tot*ones(1,N);


TH=zeros(1,N);
HI=zeros(1,N);
% % localization uncertainy
% r=[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1];
% uz=[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]/2;
r=0.05*ones(1,N);
uz=0.05*ones(1,N);
D=delta;
% 
% X=X(1:N); Y=Y(1:N); Z=Z(1:N); TH=TH(1:N); HI=HI(1:N); 
% r=r(1:N); uz=uz(1:N);

% X=ones(1,N)+2*rand(1,N)-0.5;Y=ones(1,N)+rand(1,N)-0.3;Z=ones(1,N)+rand(1,N);
% TH=zeros(1,N);HI=zeros(1,N);
% r=ones(1,N)*0.1;
% uz=ones(1,N)*0.05;

% X=rand(1,N)*0.8;
% Y=rand(1,N)*2;
% 
% Z = rand(1,N)*(zmax_tot-zmin_tot)+zmin_tot;
% TH = zeros(1,N);
% HI=zeros(1,N);
% D=delta;
% r=ones(1,N)*0.05;
% uz=ones(1,N)*0.05;

% real_r=[0.3 0.3 0.3 0.3 0.3 0.3 ];
% real_uz=[0.2 0.2 0.2 0.2 0.2 0.2];

% real_X=X;
% real_Y=Y;
% real_Z=Z;
% 
% for ki=1:length(X)
%     real_X(ki)=X(ki)+((-1)^(round(rand)))*real_r(ki)*rand(1);
%     real_Y(ki)=Y(ki)+((-1)^(round(rand)))*real_r(ki)*rand(1);
%     real_Z(ki)=Z(ki)+((-1)^(round(rand)))*real_uz(ki)*rand(1);
% end

% delta=[0.261799387799149,0.349065850398866,0.314159265358979];
% X=[0.235550712841847,0.503252238220062,0.774591052149943];
% Y = [0.244687128525606,0.523890069374973,0.251647349054111];
% Z = [1.74150792551783,1.00033593350729,1.20008080175411];
% TH = [0 0 0 ];
% HI=[0 0 0];
% D=delta;
% % localization uncertainy
% r=[0.0500    0.0800    0.0740];
% uz=[0 0 0];

% %half the angle of conic field of view
% delta=[deg2rad(20) deg2rad(20) deg2rad(20)];
% X = [3 4 3.5];
% Y = [3 3 3];
% Z = [1.2 0.9 1.5];
% TH = [0 0 0];
% HI=[0 0 0];
% D=delta;
% % localization uncertainy
% r=[0.1 0.1 0.1];
% uz=[0.3 0.3 0.3];

% %half the angle of conic field of view
% delta=[deg2rad(20)];
% 
% X = [1];
% Y = [1];
% Z = [1];
% TH = [0.5];
% HI=[0];
% D=delta;
% % localization uncertainy
% r=[0.1];
% uz=[0];

% delta=deg2rad(30);
% X = [0];
% Y = [0 ];
% Z = [1];
% TH = [0];
% HI=[0];
% D=delta;
% % %localization uncertainy
% r=[0];
% uz=[0];

% Number of nodes
%N = length(X);

zmin=zmin_tot+uz;
zmax=zmax_tot-uz;

%initial projection of field of view
[a,b,xc,yc] = calculate_ellipse_parameters(HI,TH,Z,D,X,Y,N);

%maximum distance of MAA from center of sensing pattern
[am,bm,xcm,ycm] = calculate_ellipse_parameters(h_lim*ones(1,N),0*ones(1,N),zmax,delta_max*ones(1,N),X,Y,N);
dmax=bm/tan(delta_max);

% ---------------- Simulation initializations ----------------

% Simulation steps
smax = floor(Tfinal/Tstep);
% Points Per Circle
PPC = 250;
% Radius for points on plots
disk_rad = 0.02;
% Vector for circle parameterization
t = linspace(0, 2*pi, PPC+1);
t = t(1:end-1); % remove duplicate last element
t = fliplr(t); % flip to create CW ordered circles


% Simulation data storage
Xs = zeros(smax, N);
Ys = zeros(smax, N);
Zs = zeros(smax, N);
THs = zeros(smax, N);
HIs = zeros(smax, N);
Ds = zeros(smax, N);

ycs = zeros(smax, N);
xcs = zeros(smax, N);
as = zeros(smax, N);
bs = zeros(smax, N);

cov_area = zeros(smax,1);
% cov_arear = zeros(smax,1);
H = zeros(smax,1);
% Hr = zeros(smax,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%quadcopter data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x=X;
y=Y;
z=Z;

phi=zeros(1,N);
psi=zeros(1,N);
theta=zeros(1,N);
phid=zeros(1,N);
psid=zeros(1,N);
thetad=zeros(1,N);

%gravity accelaration
g = 9.780318;
m=0.75;

A=[  0     1     0     0     0     0
     0     0     0     0     0     0
     0     0     0     1     0     0
     0     0     0     0     0     0
     0     0     0     0     0     1
     0     0     0     0     0     0];
 
B=[  0     0     0
     1     0     0
     0     0     0
     0     1     0
     0     0     0
     0     0     1  ];



R=eye(3);Q=eye(6);
%R=0.0001*eye(3);Q=5*eye(6);
[P,L,K]=care(A,B,Q,R); %RICCATI
K1=K(1,1);K2=K(1,2);tss=10/K2;
% % 
%  K(1,2)=0.99;
%  K(2,4)=0.99;
%  K(3,6)=0.99;

ud_prev=zeros(N,3);
e_x=[];
e_y=[];
e_z=[];
ex=zeros(1,N);
ey=zeros(1,N);
ez=zeros(1,N);

e_phi=[];
e_theta=[];
e_psi=[];
ephi=zeros(1,N);
etheta=zeros(1,N);
epsi=zeros(1,N);

ro=0.2;
K2=10/(ro*min(tss)); K1=power(K2/2,0.5);
K_r=[K1 K2 0 0 0 0; 0 0 K1 K2 0 0; 0 0 0 0 K1 K2];

rx=zeros(1,N);
ry=zeros(1,N);
rz=zeros(1,N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize (cell) arrays
% Coverage quality
f = zeros(1, N);
fr = zeros(1, N);

% Sensing disks
C = cell([1 N]);
% Cr = cell([1 N]);
Cgs = cell([1 N]);
Cd = cell([1 N]);
Ctemp = cell([1 N]);
C_maxz = cell([1 N]);
C_minz = cell([1 N]);

% Sensed space partitioning
W = cell([1 N]);
% Wr = cell([1 N]);
% Control inputs
uX = zeros(1,N);
uY = zeros(1,N);
uZ = zeros(1,N);
uTH = zeros(1,N);
uHI = zeros(1,N);
uD = zeros(1,N);

% Create sim struct that contains all information, used for plots
sim = struct;
sim.region = region;
sim.axis = axis_scale;
sim.PPC = PPC;
sim.zmin = zmin;
sim.zmax = zmax;
sim.X = X;
sim.Y = Y;
sim.Z = Z;
sim.theta=theta;
sim.phi=phi;
sim.psi=psi;
% sim.real_X = real_X;
% sim.real_Y = real_Y;
% sim.real_Z = real_Z;
sim.TH = TH;
sim.HI = HI;
sim.D = D;
sim.delta=delta;
sim.a = a;
sim.b = b;
sim.xc = xc;
sim.yc = yc;

sim.dmax=dmax;
sim.N = N;
sim.C = C;
sim.Cd = Cd;
sim.C_maxz=C_maxz;
sim.C_minz=C_minz;
sim.Cgs = Cgs;
sim.W = W;
sim.f = f;

sim.PLOT_STATE_3D = PLOT_STATE_3D;
sim.PLOT_STATE_2D = PLOT_STATE_2D;
sim.PLOT_STATE_PHI = 0;
sim.PLOT_STATE_QUALITY = PLOT_STATE_QUALITY;
sim.SAVE_PLOTS = SAVE_PLOTS;
check_resilience=0;

%%%%%%%%%%%%%%%%%%% Simulation %%%%%%%%%%%%%%%%%%%
if PLOT_STATE_3D || PLOT_STATE_2D || PLOT_STATE_QUALITY
	f1=figure;
%     f2=figure;
end
tic;
for s=1:smax
	fprintf('%.2f%% complete\n',100*s/smax);
    
    if check_resilience==1
        if s==smax/2
            Xs1=Xs(1:s-1,:);
            Ys1=Ys(1:s-1,:);
            Zs1=Zs(1:s-1,:);
            TH1=THs(1:s-1,:);
            HI1=HIs(1:s-1,:);
            C1=C;
            W1=W;
            yc1=ycs(1:s-1,:);
            xc1=xcs(1:s-1,:);
            as1=as(1:s-1,:);
            bs1=bs(1:s-1,:);
            
            ycds1= ycds(1:s-1,:);
            xcds1= xcds(1:s-1,:);
            ads1= ads(1:s-1,:);
            bds1= bds(1:s-1,:);
            
            [N,X,Y,Z,TH,HI,Xs,Ys,Zs,THs,HIs,ycs,xcs,as,bs,C,W,uX,uY,uZ,uz,uTH,uHI,uD,xcd,ycd,ad,bd,ycds,xcds,ads,bds,D,Ds]=agent_failure(smax,N,X,Y,Z,TH,HI,Xs,Ys,Zs,THs,HIs,ycs,xcs,as,bs,C,W,uX,uY,uZ,uz,uTH,uHI,uD,xcd,ycd,ad,bd,ycds,xcds,ads,bds,D,Ds);
        end
    end
    % ----------------- Partitioning -----------------
    
    %calculate ellipse parameters for D
    [a,b,xc,yc] = calculate_ellipse_parameters(HI,TH,Z,D,X,Y,N);
    
%     %calculate ellipse parameters for D
%     [ar,br,xcr,ycr] = calculate_ellipse_parameters(HI,TH,real_Z,D,real_X,real_Y,N);
    
    %calculate ellipse parameters for D with maximum Z
    [a_maxz,b_maxz,xc_maxz,yc_maxz] = calculate_ellipse_parameters(HI,TH,Z+uz,D,X,Y,N);
        
    %calculate ellipse parameters for D with least Z
    [a_minz,b_minz,xc_minz,yc_minz] = calculate_ellipse_parameters(HI,TH,Z-uz,D,X,Y,N);
    
    %calculate ellipse parameters for delta
    [ad,bd,xcd,ycd] = calculate_ellipse_parameters(HI,TH,Z,delta,X,Y,N);
    
    
    % Coverage quality
    for i=1:N
		f(i)=fu(Z(i)+uz(i),HI(i),D(i), zmin(i), dmax(i),delta_min,delta_max);
    end

    % Sensing patterns
    parfor i=1:N
        %calculate pattern for new zoom angle D for Z
        C{i} = ROT([(a(i)-r(i))*cos(t); (b(i)-r(i))*sin(t)],TH(i))+ ([xc(i);yc(i)]*ones(1,length(t)));
        
%         %calculate pattern for real position
%         Cr{i} = ROT([(ar(i))*cos(t); (br(i))*sin(t)],TH(i))+ ([xcr(i);ycr(i)]*ones(1,length(t)));
        
        %calculate pattern for original zoom angle delta for Z
        Cd{i} = ROT([(ad(i)-r(i))*cos(t); (bd(i)-r(i))*sin(t)],TH(i))+ ([xcd(i);ycd(i)]*ones(1,length(t)));
        
        %for localization uncertainty
        C_maxz{i} = ROT([(a_maxz(i)-r(i))*cos(t); (b_maxz(i)-r(i))*sin(t)],TH(i))+ ([xc_maxz(i);yc_maxz(i)]*ones(1,length(t)));
        C_minz{i} = ROT([(a_minz(i)-r(i))*cos(t); (b_minz(i)-r(i))*sin(t)],TH(i))+ ([xc_minz(i);yc_minz(i)]*ones(1,length(t)));
        [xo,yo]=polybool('intersection',C_maxz{i}(1,:),C_maxz{i}(2,:),C_minz{i}(1,:),C_minz{i}(2,:));
        [xo,yo]=polybool('intersection',xo,yo,Xb,Yb);
        Cgs{i}=[xo;yo];
%         if ~isempty(xo)
%             Cgs{i}=[xo(1:length(t));yo(1:length(t))];
%         else
%             Cgs{i}=[xo;yo];
%         end
        
        
    end

    % Store simulation data
    Xs(s,:) = X;
    Ys(s,:) = Y;
    Zs(s,:) = Z;
    phis(s,:) = phi;
    thetas(s,:) = theta;
    psis(s,:) = psi;
    phids(s,:)=phid;
    thetads(s,:)=thetad;
    psids(s,:)=psid;
%     real_Xs(s,:) = real_X;
%     real_Ys(s,:) = real_Y;
%     real_Zs(s,:) = real_Z;
	THs(s,:) = TH;
    HIs(s,:) = HI;
    Ds(s,:) = D;
    ycs(s,:) = yc;
    xcs(s,:) = xc;
    as(s,:) = a;
    bs(s,:) = b;
    ycds(s,:) = ycd;
    xcds(s,:) = xcd;
    ads(s,:) = ad;
    bds(s,:) = bd;
    
    % Sensed space partitioning
    for i=1:N
		% Find the cell of each node i based on all other nodes
		W{i} = sensed_partitioning_uniform_anisotropic_cell(region,Cgs, f, i,N);
    end
    
%     for i=1:N
% 		% Find the cell of each node i based on all other nodes 
%         %for real position
%         fr(i)=fu(real_Z(i),HI(i),D(i), zmin(i), dmax(i),delta_min,delta_max);
% 		Wr{i} = sensed_partitioning_uniform_anisotropic_cell(region,Cr, fr, i,N);
%     end
    
    % ----------------- Plots -----------------
    sim.X = X;
    sim.Y = Y;
    sim.Z = Z;
    sim.theta=theta;
    sim.phi=phi;
    sim.psi=psi;
%     sim.real_X = real_X;
%     sim.real_Y = real_Y;
%     sim.real_Z = real_Z;
%     sim.ar = ar;
%     sim.br = br;
%     sim.xcr = xcr;
%     sim.ycr = ycr;
	sim.TH = TH;
    sim.HI = HI;
    sim.D = D;
    sim.a = a;
    sim.b = b;
    sim.xc = xc;
    sim.yc = yc;
    sim.C = C;
%     sim.Cr = Cr;
    sim.Cd = Cd;
    sim.Cgs = Cgs;
    sim.C_maxz=C_maxz;
    sim.C_minz=C_minz;
    sim.W = W;
    sim.f = f;
%     sim.Wr = Wr;
%     sim.fr = fr;
    sim.s = s;
    sim.N=N;

%%%%%%%%%%%%%%% PLOTS %%%%%%%%%%%%%%%%%%
%     name=sprintf('s%i.png',s);
%     saveas(gcf,name)
%     clf(f2);
%     plot( Tstep*linspace(1,s,s), H(1:s), 'k');

     clf(f1);
     plot_sim_UAV_zoom(f1,sim);
     
% progress= sprintf('%.2f sec\n%.2f%%',s*Tstep,s*100/smax);
% text(4,4,progress)
% % waitbar(s / smax,f1)
% NAME= sprintf('video1/s%d.png',s);
% saveas(gcf,NAME);
%     
    % ----------------- Objective -----------------
    % Find covered area and H objective
    for i=1:N
        if ~isempty(W{i})
            cov_area(s) = cov_area(s) + polyarea_nan(W{i}(1,:), W{i}(2,:));
            H(s) = H(s) + f(i) * polyarea_nan(W{i}(1,:), W{i}(2,:));
        end
    end
    
%     % ----------------- Real Objective -----------------
%     % Find covered area and H objective
%     for i=1:N
%         if ~isempty(Wr{i})
%             cov_arear(s) = cov_arear(s) + polyarea_nan(Wr{i}(1,:), Wr{i}(2,:));
%             Hr(s) = Hr(s) + fr(i) * polyarea_nan(Wr{i}(1,:), Wr{i}(2,:));
%         end
%     end
    
         
    % ----------------- Control law -----------------
    parfor i=1:N % parfor faster here
        % Create anonymous functions for the Jacobians
        % The functions include parameters specific to this node
        
        Jxy = @(q) J_ellipse_xy(q);
        %jacobians for z_min possible      
			Jz_maxz = @(q) J_ellipse_z(q, X(i), Y(i), Z(i)+uz(i), TH(i), HI(i) ,a_maxz(i)-r(i), b_maxz(i)-r(i),xc_maxz(i), yc_maxz(i));
            
			Jth_maxz = @(q) J_ellipse_th(q, X(i), Y(i), Z(i)+uz(i), TH(i), HI(i) ,a_maxz(i)-r(i), b_maxz(i)-r(i),xc_maxz(i), yc_maxz(i));
            
            Jh_maxz=@(q) J_ellipse_h(q, X(i), Y(i), Z(i)+uz(i), TH(i), HI(i) ,a_maxz(i)-r(i), b_maxz(i)-r(i),xc_maxz(i), yc_maxz(i),D(i));
            
            Jd_maxz=@(q) J_ellipse_d(q, X(i), Y(i), Z(i)+uz(i), TH(i), HI(i) ,a_maxz(i)-r(i), b_maxz(i)-r(i),xc_maxz(i), yc_maxz(i),D(i));
        %jacobians for z_max possible
            Jz_minz = @(q) J_ellipse_z(q, X(i), Y(i), Z(i)-uz(i), TH(i), HI(i) ,a_minz(i)-r(i), b_minz(i)-r(i),xc_minz(i), yc_minz(i));
            
			Jth_minz = @(q) J_ellipse_th(q, X(i), Y(i), Z(i)-uz(i), TH(i), HI(i) ,a_minz(i)-r(i), b_minz(i)-r(i),xc_minz(i), yc_minz(i));
            
            Jh_minz=@(q) J_ellipse_h(q, X(i), Y(i), Z(i)-uz(i), TH(i), HI(i) ,a_minz(i)-r(i), b_minz(i)-r(i),xc_minz(i), yc_minz(i),D(i));
            
            Jd_minz=@(q) J_ellipse_d(q, X(i), Y(i), Z(i)-uz(i), TH(i), HI(i) ,a_minz(i)-r(i), b_minz(i)-r(i),xc_minz(i), yc_minz(i),D(i));
        
            %control laws
		[uX(i), uY(i)] = control_uniform_planar(region, W, Cgs,f, i, Jxy);
		uZ(i) = control_uniform_altitude(region, W, Cgs,C_maxz,f, dfuz(Z(i)+uz(i),HI(i),D(i),zmin(i),dmax(i),delta_min,delta_max), i, Jz_maxz,Jz_minz);
		uTH(i) = control_uniform_pan(region, W, Cgs,C_maxz,f, i, Jth_maxz,Jth_minz);
        uHI(i) = control_uniform_tilt(region, W, Cgs,C_maxz,f, dfuh(Z(i)+uz(i),HI(i),D(i),zmin(i),dmax(i),delta_min,delta_max), i, Jh_maxz,Jh_minz);
        uD(i) = control_uniform_zoom(region, W, Cgs,C_maxz,f, dfud(Z(i)+uz(i),HI(i),D(i),zmin(i),dmax(i),delta_min,delta_max), i, Jd_maxz,Jd_minz);

    end

    
    % Control inputs with gain
    uX = axy * uX;
    uY = axy * uY;
    uZ = az * uZ;
	uTH = ath * uTH;
    uHI = ah *uHI;
    uD = azoom *uD;
    
    % ----------------- Simulate with ode -----------------
    Tspan = [s*Tstep (s+1)*Tstep];
    IC = [X Y Z TH HI D]';
    u = [uX uY uZ uTH uHI uD]';
    [T, ode_state] = ode45(@(t,y) DYNAMICS_simple(t, y, u), Tspan, IC);
    
    % Keep the last row of XYZ
    
    Xnn = ode_state(end, 1:N );
    Ynn = ode_state(end, N+1:2*N );
    Znn = ode_state(end, 2*N+1:3*N );
    THnn = ode_state(end, 3*N+1:4*N );
    HInn=ode_state(end, 4*N+1:5*N );
    Dnn=ode_state(end, 5*N+1:6*N );

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%% insert quadrotor dynamic model %%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     parfor node=1:N  
%         
%         if s>1
%             uv=[Xs(s,node)-Xs(s-1,node),Ys(s,node)-Ys(s-1,node),Zs(s,node)-Zs(s-1,node)]*1/Tstep;            
%         else
%             uv=[0 0 0];
%         end
%         
%         a_d=[uX(node)-ud_prev(node,1),uY(node)-ud_prev(node,2),uZ(node)-ud_prev(node,3)];
%         
%         if s>2
%             a=1/Tstep*[uv(1)-(1/Tstep*(Xs(s-1,node)-Xs(s-2,node))),uv(2)-(1/Tstep*(Ys(s-1,node)-Ys(s-2,node))),uv(3)-(1/Tstep*(Zs(s-1,node)-Zs(s-2,node)))];
%         else
%             if s==1
%                 a=[0 0 0];
%             else
%                 a=uv/Tstep;
%             end
%         end
%         [Xn(node),Yn(node),Zn(node),e,rx(node),ry(node),rz(node)]=quadrotor(s,Tstep,A,B,K,Xnn(node),Ynn(node),Znn(node),Xs(s,node),Ys(s,node),Zs(s,node),uX(node),uY(node),uZ(node),uv(1),uv(2),uv(3),a_d(1),a_d(2),a_d(3),rx(node),ry(node),rz(node));       
%         %[Xn(node),Yn(node),Zn(node),e,rx(node),ry(node),rz(node)]=quadrotor(s,Tstep,A,B,K,Xn(node),Yn(node),Zn(node),Xs(s,node),Ys(s,node),Zs(s,node),uX(node),uY(node),uZ(node),uv(1),uv(2),uv(3),a_d(1),a_d(2),a_d(3),a(1),a(2),a(3));
%         ex(node)=e(1);
%         ey(node)=e(2);
%         ez(node)=e(3);
%         
% %        display(rx(node));display(ry(node));
%         phid(node)=atan2(-ry(node),(rz(node)+g));
%         thetad(node)=atan2(rx(node)*cos(phid(node)),(rz(node)+g));
%         psid(node)=atan(uY(node)/uX(node));
% 
%         %psid(node)=0;
% 
%         
%         if s>1
%             ur=[phis(s,node)-phis(s-1,node),thetas(s,node)-thetas(s-1,node),psis(s,node)-psis(s-1,node)]*1/Tstep;   
%             ud=[phid(node)-phids(s-1,node),thetad(node)-thetads(s-1,node),psid(node)-psids(s-1,node)]*1/Tstep;   
%         else
%             ur=[phis(s,node),thetas(s,node),psis(s,node)]*1/Tstep;
%             ud=[phids(s,node),thetads(s,node),psids(s,node)]*1/Tstep;
%         end
%         
%         if s>2
%             a=1/Tstep*[ur(1)-(1/Tstep*(phis(s-1,node)-phis(s-2,node))),ur(2)-(1/Tstep*(thetas(s-1,node)-thetas(s-2,node))),ur(3)-(1/Tstep*(psis(s-1,node)-psis(s-2,node)))];
%             a_d=1/Tstep*[ud(1)-(1/Tstep*(phids(s-1,node)-phids(s-2,node))),ud(2)-(1/Tstep*(thetads(s-1,node)-thetads(s-2,node))),ud(3)-(1/Tstep*(psids(s-1,node)-psids(s-2,node)))];
%         else
%             if s==1
%                 a=1/Tstep*[ur(1),ur(2),ur(3)];
%                 a_d=1/Tstep*[ud(1),ud(2),ud(3)];
%             else
%                 a=1/Tstep*[ur(1)-(1/Tstep*(phis(s-1,node))),ur(2)-(1/Tstep*(thetas(s-1,node))),ur(3)-(1/Tstep*(psis(s-1,node)))];
%                 a_d=1/Tstep*[ud(1)-(1/Tstep*(phids(s-1,node))),ud(2)-(1/Tstep*(thetads(s-1,node))),ud(3)-(1/Tstep*(psids(s-1,node)))];
%             end
%         end
%         
%        [phi(node),theta(node),psi(node),e_angles]=orientation(s,Tstep,A,B,K_r,phid(node),thetad(node),psid(node),phis(s,node),thetas(s,node),psis(s,node),uX(node),uY(node),uZ(node),u(1),u(2),u(3),a_d(1),a_d(2),a_d(3),a(1),a(2),a(3));
%        ephi(node)=e_angles(1);
%        etheta(node)=e_angles(2);
%        epsi(node)=e_angles(3);
%     %display(phi(node));
%     end
%         
%     e_x=[e_x;ex];
%     e_y=[e_y;ey];
%     e_z=[e_z;ez];
%     
%     e_phi=[e_phi;ephi];
%     e_theta=[e_theta;etheta];
%     e_psi=[e_psi;epsi];
%     
%     phids(s,:)=phid;
%     thetads(s,:)=thetad;
%     psids(s,:)=psid;
%     
%     phis(s,:)=phi;
%     thetas(s,:)=theta;
%     psis(s,:)=psi;
%     
%     ud_prev=[uX',uY',uZ'];
%     
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    parfor i=1:N
        sense=min_distance_point(Xnn(i),Ynn(i),Znn(i),THnn(i),HInn(i),Dnn(i),t,r(i));
        
        
%--------------------- Zoom control

        if (Dnn(i)>=delta_min & Dnn(i)<=delta_max)
            D(i)=Dnn(i);
        else
            if Dnn(i)>delta_max
                D(i)=delta_max;
            else
                D(i)=delta_min;
            end                
        end

%--------------------- Position control
         if (inpolygon(Xnn(i),Ynn(i),Xb,Yb))
            X(i)=Xnn(i);
            Y(i)=Ynn(i);  
%             
%             real_X(i) = Xn(i)+((-1)^(round(rand)))*real_r(i)*rand(1);
%             real_Y(i) = Yn(i)+((-1)^(round(rand)))*real_r(i)*rand(1);
         end
%          X(i)=Xnn(i);
%          Y(i)=Ynn(i);  
%         

%--------------------- Altitude control
        if (sum(inpolygon(sense(1,:),sense(2,:),Xb,Yb))>=1)
            
            if (Znn(i)<=zmax(i) & Znn(i)>=zmin(i))
                Z(i)=Znn(i);   
            end
            if Znn(i)>zmax(i)
                Z(i)=zmax(i);  
            end
            if Znn(i)<zmin(i)
                Z(i)=zmin(i);  
            end
%             real_Z(i) = Z(i)+((-1)^(round(rand)))*real_r(i)*rand(1);
 

%--------------------- Pan control
            TH(i)=THnn(i);
            

%--------------------- Tilt control
            if (abs(HInn(i))<=deg2rad(h_lim))
               HI(i)=HInn(i); 
            end
            if HInn(i)>deg2rad(h_lim)
               HI(i)=deg2rad(h_lim);
            end
            
        end
        
    end
    
    end

elapsed_time = toc;
average_iteration = elapsed_time / smax;
fprintf('\nSimulation time: %.4f s\n', elapsed_time)
fprintf('Average iteration time: %.4f s\n', average_iteration)



%%%%%%%%%%%%%%%%%%% Final plots %%%%%%%%%%%%%%%%%%%
% hold on;
% plot3(Xs,Ys,Zs,'b');
% 
% 
% v = VideoWriter(video_name);
% v.FrameRate = 4/Tstep;
% open(v)
% for i=1:smax
% nam=sprintf('video1/s%i.png',i);
% yy=imread(nam);
% writeVideo(v,yy)
% end
% close(v)

% Plot covered area
figure;
plot( Tstep*linspace(1,s,s), 100*cov_area(1:s)/region_area, 'k');
% plot( Tstep*linspace(1,s,s), 100*cov_area(1:s)/region_area, '--r');
% legend('virtual locations','real locations');
axis([0 Tstep*smax 0 100]);
h = xlabel('$Time ~(s)$');
set(h,'Interpreter','latex')
h = ylabel('$A_{cov}~(\%)$');
set(h,'Interpreter','latex')

% figure;
% subplot(3,1,1);
% plot( Tstep*linspace(1,s,s), (e_x));
% grid on;
% ylabel('x_{error}');
% subplot(3,1,2);
% plot( Tstep*linspace(1,s,s), (e_y));
% ylabel('y_{error}');
% grid on;
% subplot(3,1,3);
% plot( Tstep*linspace(1,s,s), (e_z));
% ylabel('z_{error}');
% grid on;
% h = xlabel('$Time ~(s)$');
% set(h,'Interpreter','latex')
% 
% 
% figure;
% subplot(3,1,1);
% plot( Tstep*linspace(1,s,s), rad2deg(e_phi));
% grid on;
% ylabel('\phi_{error}');
% subplot(3,1,2);
% plot( Tstep*linspace(1,s,s), rad2deg(e_theta));
% ylabel('\theta_{error}');
% grid on;
% subplot(3,1,3);
% plot( Tstep*linspace(1,s,s), rad2deg(e_psi));
% ylabel('\psi_{error}');
% grid on;
% h = xlabel('$Time ~(s)$');
% set(h,'Interpreter','latex')


% figure;
% subplot(3,1,1);
% plot( Tstep*linspace(1,s,s), rad2deg(phids+e_phi));
% grid on;
% ylabel('\phi');
% subplot(3,1,2);
% plot( Tstep*linspace(1,s,s), rad2deg(thetads+e_theta));
% ylabel('\theta');
% grid on;
% subplot(3,1,3);
% plot( Tstep*linspace(1,s,s), rad2deg(psids+e_psi));
% ylabel('\psi');
% grid on;
% h = xlabel('$Time ~(s)$');
% set(h,'Interpreter','latex')


% Plot objective
figure;
%hold on;
plot( Tstep*linspace(1,s-1,s-1),H(1:s-1), 'k');

% hold on;
% plot( Tstep*linspace(1,s-160,s-160), H(1:s-160), '--r');

% plot( Tstep*linspace(1,s,s), H(1:s), 'r');
% legend('Voronoi-free','Voronoi');
h = xlabel('$Time ~(s)$');
set(h,'Interpreter','latex')
h = ylabel('$\mathcal{H}$');
set(h,'Interpreter','latex')

% figure;
% sim.X=sim.real_X;
% sim.Y=sim.real_Y;
% sim.Z=sim.real_Z;
% sim.C=sim.Cr;
% sim.W=sim.Wr;
% sim.a = sim.ar;
% sim.b =sim.br;
% sim.xc =sim.xcr;
% sim.yc =sim.ycr;
% plot_sim_UAV_zoom(sim);

% Save trajectories
traj = zeros(5,smax,N);
traj(1,:,:) = Xs;
traj(2,:,:) = Ys;
traj(3,:,:) = Zs;
traj(4,:,:) = THs;
traj(5,:,:) = HIs;
%%%%%%%%%%%%%%%%%%% Save Results %%%%%%%%%%%%%%%%%%%
if SAVE_RESULTS
    filename = ...
        strcat( 'results_uniform_anisotropic_', ...
        datestr(clock,'yyyymmdd_HHMM') , '.mat' );
    save(filename);
end











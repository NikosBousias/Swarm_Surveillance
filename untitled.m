
%fill([-0.2 4.5 4.5 -0.2],[3.2 3.2 -0.2 -0.2],[0.85 0.9 0.9])
fill(sim.region(1,:),sim.region(2,:),[0.85 0.9 0.9])
alpha(0.4)
hold on;
plot(sim.region(1,:),sim.region(2,:),'--k')
hold on;
plot([region(1,1),region(1,end)],[sim.region(2,1),sim.region(2,end)],'--k')
%axis([-1 6 -0.5 3.5])
axis([-0.5 5.5 -0.5 5.5]);
plot(Xs(1,:),Ys(1,:),'ro')

plot(Xs(s,:),Ys(s,:),'go')

plot(Xs,Ys)

h=text(3.2,3.8,'$\Omega \equiv \hat{\Omega}$');
set(h,'Interpreter','latex')
%h=text(4,0,'$\hat{\Omega}$');
set(h,'Interpreter','latex')

grid on;
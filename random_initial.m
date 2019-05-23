function [x,y]= random_initial(region,k)

reg=length(region(1,:));
x=region(1,:);
y=region(2,:);


for i=1:k
    r1=randi([1 length(x)]);
    r2=randi([1 length(x)]);
    while r2==r1
        r2=randi([1 length(x)]);
    end
    
    r3=randi([1 length(x)]);
    while r3==r1 || r3==r2
        r2=randi([1 length(x)]);
    end
    
    x=[x mean([x(r1),x(r2),x(r3)])];
    y=[y mean([y(r1),y(r2),y(r3)])];
end

x=x(reg:end);
y=y(reg:end);

end
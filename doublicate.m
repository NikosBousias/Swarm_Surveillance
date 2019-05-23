function [ l ] = doublicate( p )
l=[];
for i=1:length(p)
    l=[l p(i) p(i)];    
end

end


function out = mapFeature(X1, X2)

power = 6;
out = ones(size(X1(:,1)));
for i = 1:power
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end 
end

end


function [bestEpsilon bestF1] = selectThreshold(yval, pval)

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    predictions = pval < epsilon;
    
    tp = sum(yval .* predictions);
    fp = sum((yval == 0) .* predictions);
    fn = sum(yval .* (predictions == 0));
    
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);

    F1 = (2 * prec * rec) / (prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end

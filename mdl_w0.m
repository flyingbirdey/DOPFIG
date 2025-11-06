function [optimal_order, optimal_mdl_value, optimal_y_fit,optimal_xishu,optimal_sigma] = mdl_w0(x,y,max_order)

n = length(y); 
min_mdl = Inf; 
optimal_order = 0; 
mdl_values = zeros(1, max_order+1); 
error_terms = zeros(1, max_order+1); 
penalty_terms = zeros(1, max_order+1); 
p = zeros(max_order+1,max_order+1); 
y_fit = zeros(n,max_order+1); 
residuals = zeros(n, max_order+1);
f = zeros(n, max_order+1); 


for k = 0:max_order
    p(1:k+1,k+1) = polyfit(x, y, k); 
    y_fit(:, k+1) = polyval(p(1:k+1,k+1), x);
    residuals(:,k+1) = y - y_fit(:,k+1); 

    rss(k+1) = sum(residuals(:,k+1).^2); 
    sigma_squared(k+1) = rss(k+1) / n; 
end

sigma_med = median(sigma_squared); 
sigma_global = sigma_med; 
sigma_max = max(sigma_squared); 
sigma_min = min(sigma_squared); 

% 计算各阶MDL值
for i = 0:max_order
    w(i+1) = 1 - ((sigma_squared(i+1) - sigma_min) / (sigma_max - sigma_min));
    sigma_w(i+1) = w(i+1) * sigma_squared(i+1) + (1-w(i+1)) * sigma_global ; 

    f(:,i+1) = exp(-((residuals(:,i+1)).^2) / (2 * (sigma_w(i+1))));

    error_term = (n / 2) * log(n / sum(f(:,i+1)));

    penalty_term = ((i + 1) / 2) * log(n) ;

    mdl = error_term + penalty_term ;

    error_terms(i+1) = error_term;
    penalty_terms(i+1) = penalty_term;
    mdl_values(i+1) = mdl;

    if mdl < min_mdl 
        min_mdl = mdl; 
        optimal_order = i;
        optimal_mdl_value = mdl_values(i+1);
        optimal_y_fit = y_fit(:, i+1);
        optimal_xishu = p(:,i+1);
        optimal_sigma = sigma_squared(i+1);
    end

end

end
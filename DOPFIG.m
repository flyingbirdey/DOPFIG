%% 1.初始化，导入数据
clc; clear; close all;tic;

data0 = readtable('daily-min-temperatures.csv');
data = data0.Temp;
x = data;
n = size(x,1); 
t = (1:n)';

q = 1; 
cu_k = 4; 
cu_k_new = cu_k; 
max_order = 10;

%% 2.使用 Savitzky-Golay 滤波
sg_k = 3; 

for sg_t = (sg_k + 2) : 2 : n*0.1  
    y_sg = sgolayfilt(x, sg_k, sg_t);  

    %% 3.计算自相关函数找到主周期
    [xc,lags] = xcorr(y_sg,'coeff');  
    mid = ceil(length(xc)/2);      
    acf = xc(mid:end);             
    lag = lags(mid:end);

    %% 找自相关函数的峰值
    [pks,locs] = findpeaks(acf, lag); 

    % 峰值差分
    period = diff(locs); 
    period = [locs(1) period]; 
    cv = std(period) / mean(period); 

    if cv < 0.1 && length(period) > 2 
        break;
    end

end
%% 绘图
figure(1);
plot(t, x, 'color', [0.7 0.7 0.7]); hold on; 
plot(t, y_sg, 'r', 'LineWidth', 2); 

legend('原始信号', 'SG滤波后');
xlabel('时间'); ylabel('因变量');
title('Savitzky-Golay 滤波效果');
grid on;

%% 稳健斜率估计,确定主周期
slopes = [];

for j = 2:length(locs)
    slopes(end+1) = (locs(j) - locs(1)) / (j - 1);
end

period_main = round(median(slopes));

fprintf('首点锚定周期估计: %.2f\n', period_main);

%% 4.初始化分段拟合
d = floor(n / period_main); 
yu = mod(n , period_main); 
opt_order = zeros(1, d); 
opt_mdl_value = zeros(1, d); 
opt_y_fit = zeros(period_main,d); 
opt_xishu = zeros(max_order+1,d); 
opt_sigma = zeros(1, d); 

%% 5.利用FMDL找每一段的最优阶数，并拟合绘图
period0 = repmat(period_main, 1, d); 
T = period0; 
locs_t = yu + period_main * (1:d); 

figure(2);
subplot(2,1,1);
plot(t,y_sg); title('sg后时间序列');
hold on
plot(locs_t,y_sg(locs_t),'ro'); 
xlabel('t'); ylabel('x');

subplot(2,1,2);
plot(lag,acf,'b','LineWidth',1.5); hold on;
plot(locs,pks,'ro'); 
title('自相关函数 (ACF)');
xlabel('滞后'); ylabel('自相关系数');
legend('ACF','峰值');

figure(3);
plot(1:n, x, 'b.', 'DisplayName', '原始数据'); 

y = zeros(period_main,d); 
x0 = zeros(period_main,d); 
x_ni = 1:period_main; 

for i=1:d
    y(:,i) = x(((i-1)*period_main+1+yu):i*period_main+yu);
    x0(:,i) = ((i-1)*period_main+1+yu):i*period_main+yu;
    [opt_order(i), opt_mdl_value(i), opt_y_fit(:,i), opt_xishu(:,i), opt_sigma(i)] = mdl_w0(x_ni,y(:,i),max_order);
    plot(x0(:, i), opt_y_fit(:, i), 'r-', 'LineWidth', 2);
    grid on;
    title('原始数据与最优阶多项式拟合结果');
    xlabel('x'); ylabel('y');
end

disp(['最优多项式阶数为: ', num2str(opt_order)]);

opt_xishu1 = zeros(max(opt_order)+1,d);
for k = 1:d
    f = flip(opt_xishu(1:max(opt_order)+1,k));
    opt_xishu1(1:opt_order(k)+1,k) = f(max(opt_order)-opt_order(k)+1:end);
end

x_ci = zeros(period_main,max(opt_order)+1);
for p = 1:max(opt_order)+1
    x_ci(:,p) = (1:period_main).^(p-1);
end


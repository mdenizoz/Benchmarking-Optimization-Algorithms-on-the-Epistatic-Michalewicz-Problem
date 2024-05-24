clear all
close all
clc

% Mesh grid oluşturma
X1 = -2:0.01:2;
X2 = -2:0.01:2;
[x1, x2] = meshgrid(X1, X2);
F = 0.25 * x1.^4 - 0.5 * x1.^2 + 0.1 * x1 + 0.5 * x2.^2;
realFMin = min(min(F));

% Her algoritmanın yolunu saklamak için cell array
paths = cell(5, 1);

%% Newton-Raphson
fprintf('Newton-Raphson Algorithm\n');
x = [-1; 1];
epsilon = 1e-4;
max_iter = 10;

% Newton-Raphson yolu
newton_path = x';

grad = gradfunc(x);
H = hessianfunc(x);
x_next = x - inv(H) * grad;
newton_path = [newton_path; x_next'];
k = 3;
fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x));
tic;
while norm(gradfunc(x_next)) > epsilon && k <= max_iter
    x = x_next;
    grad = gradfunc(x);
    H = hessianfunc(x);
    x_next = x - inv(H) * grad;
    newton_path = [newton_path; x_next'];
    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs. error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)));
    k = k + 1;
end
elapsed_time = toc;
fprintf('Elapsed time: %f seconds\n', elapsed_time);

paths{1} = newton_path;

%% Hestenes-Stiefel Algorithm
fprintf('Hestenes-Stiefel Algorithm\n');
x = [-1; 1];
epsilon = 1e-4;
max_iter = 10;

% Hestenes-Stiefel yolu
hs_path = x';

grad = gradfunc(x);
d = -grad;
alpha = 1;
x_next = x + alpha * d;
hs_path = [hs_path; x_next'];
k = 3;
fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x));
tic;
while norm(gradfunc(x_next)) > epsilon && k <= max_iter
    x = x_next;
    grad_new = gradfunc(x);
    beta = (grad_new' * (grad_new - grad)) / (d' * (grad_new - grad));
    d = -grad_new + beta * d;
    x_next = x + alpha * d;
    grad = grad_new;
    hs_path = [hs_path; x_next'];
    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs. error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)));
    k = k + 1;
end
elapsed_time = toc;
fprintf('Elapsed time: %f seconds\n', elapsed_time);

paths{2} = hs_path;

%% Polak-Ribiere Algorithm
fprintf('Polak-Ribiere Algorithm\n');
x = [-1; 1];
epsilon = 1e-4;
max_iter = 10;

% Polak-Ribiere yolu
pr_path = x';

grad = gradfunc(x);
d = -grad;
alpha = 1;
x_next = x + alpha * d;
pr_path = [pr_path; x_next'];
k = 3;
fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x));
tic;
while norm(gradfunc(x_next)) > epsilon && k <= max_iter
    x = x_next;
    grad_new = gradfunc(x);
    beta = (grad_new' * (grad_new - grad)) / (grad' * grad);
    d = -grad_new + beta * d;
    x_next = x + alpha * d;
    grad = grad_new;
    pr_path = [pr_path; x_next'];
    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs. error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)));
    k = k + 1;
end
elapsed_time = toc;
fprintf('Elapsed time: %f seconds\n', elapsed_time);

paths{3} = pr_path;

%% Fletcher-Reeves Algorithm
fprintf('Fletcher-Reeves Algorithm\n');
x = [-1; 1];
epsilon = 1e-4;
max_iter = 10;

% Fletcher-Reeves yolu
fr_path = x';

grad = gradfunc(x);
d = -grad;
alpha = 1;
x_next = x + alpha * d;
fr_path = [fr_path; x_next'];
k = 3;
fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x));
tic;
while norm(gradfunc(x_next)) > epsilon && k <= max_iter
    x = x_next;
    grad_new = gradfunc(x);
    beta = (grad_new' * grad_new) / (grad' * grad);
    d = -grad_new + beta * d;
    x_next = x + alpha * d;
    grad = grad_new;
    fr_path = [fr_path; x_next'];
    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs. error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)));
    k = k + 1;
end
elapsed_time = toc;
fprintf('Elapsed time: %f seconds\n', elapsed_time);

paths{4} = fr_path;

%% Quasi-Newton (BFGS) Algorithm
fprintf('Quasi-Newton Algorithm\n');
x = [-1; 1];
epsilon = 1e-4;
max_iter = 10;

% Quasi-Newton yolu
qn_path = x';

H = eye(2);
grad = gradfunc(x);
k = 1;
x_next = x - H * grad;
qn_path = [qn_path; x_next'];
fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x));
tic;
while norm(gradfunc(x_next)) > epsilon && k <= max_iter
    s = x_next - x;
    y = gradfunc(x_next) - grad;
    rho = 1 / (y' * s);
    H = (eye(2) - rho * s * y') * H * (eye(2) - rho * y * s') + rho * (s * s');
    x = x_next;
    grad = gradfunc(x);
    x_next = x - H * grad;
    qn_path = [qn_path; x_next'];
    fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs. error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)));
    k = k + 1;
end
elapsed_time = toc;
fprintf('Elapsed time: %f seconds\n', elapsed_time);

paths{5} = qn_path;

% Contour plot of all paths
figure;
contourf(x1, x2, F);
hold on;
colors = {'r', 'g', 'b', 'c', 'm'};
labels = {'Newton-Raphson', 'Hestenes-Stiefel', 'Polak-Ribiere', 'Fletcher-Reeves', 'Quasi-Newton'};
for i = 1:length(paths)
    path = paths{i};
    plot(path(:, 1), path(:, 2), [colors{i} '-*'], 'DisplayName', labels{i});
end
title('Optimization Paths on Contour Plot');
xlabel('x1');
ylabel('x2');
legend show;
set(gca, 'fontsize', 12);

% 3D plot of all paths
figure;
mesh(x1, x2, F);
hold on;
for i = 1:length(paths)
    path = paths{i};
    z_vals = arrayfun(@(j) func(path(j, :)'), 1:size(path, 1));
    plot3(path(:, 1), path(:, 2), z_vals, 'Color', colors{i}, 'LineWidth', 2, 'DisplayName', labels{i});
end
title('3D Optimization Paths');
xlabel('x1');
ylabel('x2');
zlabel('f(x)');
legend show;
grid on;
set(gca, 'fontsize', 12);

% Individual contour plots for each algorithm
figure;
for i = 1:length(paths)
    subplot(2, 3, i);
    contourf(x1, x2, F);
    hold on;
    path = paths{i};
    plot(path(:, 1), path(:, 2), [colors{i} '-*']);
    title([labels{i} ' Algorithm']);
    xlabel('x1');
    ylabel('x2');
    set(gca, 'fontsize', 12);
end

% Fonksiyon tanımlamaları
function val = func(x)
    val = 0.25 * x(1)^4 - 0.5 * x(1)^2 + 0.1 * x(1) + 0.5 * x(2)^2;
end

function grad = gradfunc(x)
    grad = [x(1)^3 - x(1) + 0.1; x(2)];
end

function H = hessianfunc(x)
    H = [3 * x(1)^2 - 1, 0; 0, 1];
end
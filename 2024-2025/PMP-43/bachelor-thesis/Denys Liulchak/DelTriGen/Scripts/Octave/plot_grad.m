function plot_grad()
    pkg('load', 'symbolic');

    x = sym('x');
    y = sym('y');
    f = (x + y) .* (x .^ 2 + y .^ 2 - 1);
%    f = -5 / (x .^ 2 + y .^ 2 + 1);

    fx = function_handle(diff(f, x), 'vars', { x, y });
    fy = function_handle(diff(f, y), 'vars', { x, y });

    n  = 101;
    xv = linspace(-5, 5, n);
    yv = linspace(-5, 5, n);

    step = 5;
    idx  = 1: step: n;

    [X, Y]    = meshgrid(xv, yv);
    U         = fx(X, Y);
    V         = fy(X, Y);
    grad_norm = sqrt(U .^ 2 + V .^ 2);



    field_h = figure('Name', 'Gradient Field', 'NumberTitle', 'off');

    quiver(X(idx, idx), Y(idx, idx), U(idx, idx), V(idx, idx), 0.9, ...
          'LineWidth', 1.5, 'Color', 'blue');

    set(gca(), 'FontSize', 25);

    axis('equal');
    grid('on');

    xy_lim = axis();

    xlabel('x');
    ylabel('y');

    title('\nabla f');

    refresh(field_h);



    map_h = figure('Name', 'Gradient Norm', 'NumberTitle', 'off');

    contourf(X, Y, grad_norm, 'LineColor', 'none');

    set(gca(), 'FontSize', 25);

    axis(xy_lim);

    xlabel('x');
    ylabel('y');

    title('||\nabla f||');

    colormap('turbo');
    colorbar_h = colorbar('Location', 'eastoutside', 'FontSize', 25);
    ylabel(colorbar_h, 'Euclidean norm');

    refresh(map_h);
end


function QHullTest
   k = 10;
   nValues = [k, k .^ 2, k .^ 3, k .^ 4, k .^ 5, k .^ 6];

   len = length(nValues);
   times = zeros(1, len);

   for i = 1: len
       n = nValues(i);

       points = randi([1, 100000], n, 2);

       tic;
       tri = delaunay(points);
       times(i) = toc;
   end

   loglog(nValues, times, '-o', 'LineWidth', 2);
   xlabel('Кількість точок (N)');
   ylabel('Час виконання (секунди)');
   title('Залежність часу побудови тріангуляції Делоне від кількості вхідних точок');
   set(gca, 'FontSize', 25);



#   a = 3;
#   b = 1;
#
#   r = 1;
#
#   n = 1000;
#   t = linspace(0, 2 .* pi, n);
#   t(end) = [];
#
#   x = a .* r .* cos(t);
#   y = b .* r .* sin(t);
#
#   ni = 20;
#   lastIndex = n - floor(n ./ ni);
#
#   indexes = floor(linspace(1, lastIndex, ni));
#
#   xi = x(indexes);
#   yi = y(indexes);
#tic;
#   tri = delaunay(xi, yi);
#toc;
#
#   plot(x, y, 'LineStyle', '-', 'LineWidth', 2, 'Color', 'b', 'DisplayName',
#   'Область');
#
#   hold on;
#
#   triplot(tri, xi, yi, 'LineStyle', '-',
#      'LineWidth', 2, 'Color', 'g', 'Marker', 'o', 'MarkerSize', 2,
#      'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r', 'DisplayName',
#      'Тріангуляція');
#
#   hold off;
#   grid on;
#   axis equal;
#
#   set(gca, 'FontSize', 25);
#   title('Тріангуляція Делоне');
#   legend('Fontsize', 25, 'Location', 'northeast');
endfunction


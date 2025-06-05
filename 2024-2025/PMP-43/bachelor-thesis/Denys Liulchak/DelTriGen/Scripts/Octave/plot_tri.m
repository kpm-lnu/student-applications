function plot_tri(points_fpath, tri_fpath)

%         READ POINTS
%______________________________

    points_fid = open_file(points_fpath, 'r');

    coordinates_numel        = 2;
    [ points, points_numel ] = try_fscanf(points_fid, '%f', [ coordinates_numel, Inf ]);

    points        = points';
    points_numel /= coordinates_numel;

    close_file(points_fid);

%        READ TRIANGLES
%______________________________

    tri_fid = open_file(tri_fpath, 'r');

    vertices_numel                 = 3;
    [ triangles, triangles_numel ] = try_fscanf(tri_fid, '%d', [ vertices_numel, Inf ]);

    triangles        = triangles' + 1;
    triangles_numel /= vertices_numel;

    close_file(tri_fid);

%         VALIDATION
%______________________________

    if (triangles_numel == 0)
        warning('%s: empty triangulation', mfilename());

        return;
    end

    if (any(~isfinite(points(:))))
        error('%s: the coordinates of the points must be finite numbers', mfilename());
    end

    if (any(triangles(:) < 1) || ...
        any(triangles(:) > points_numel))
        error([ '%s: the indices of the vertices of the triangles ' , ...
                'must be within the range [0, %d]' ], mfilename(), points_numel - 1);
    end

%      PLOT TRIANGULATION
%______________________________

    window_h = figure('Name', 'Triangulation', 'NumberTitle', 'off');

    triplot(triangles, points(:, 1), points(:, 2), 'LineStyle', '-', ...
            'LineWidth', 1.5, 'Color', 'green', 'Marker', 'o', 'MarkerSize', 1.5, ...
            'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red');

    set(gca(), 'FontSize', 25);

    axis('equal');
    grid('on');

    xlabel('x');
    ylabel('y');

    title('Delaunay triangulation');

    refresh(window_h);
    waitfor(window_h);
end

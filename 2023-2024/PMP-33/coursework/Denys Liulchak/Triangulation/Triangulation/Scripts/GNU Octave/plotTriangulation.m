function plotTriangulation(pointsFilePath, triangulationFilePath)

#         VALIDATION
#______________________________

   ExitCodes.IOException     = -1;
   ExitCodes.Ok              = 0;
   ExitCodes.InvalidFilePath = 1;

   if !exist(pointsFilePath, 'file') || ...
      !exist(triangulationFilePath, 'file')

      printf("Invalid file path\n");
      exit(ExitCodes.InvalidFilePath, "force")
   endif

#         READ POINTS
#______________________________

   pointsFileId = fopen(pointsFilePath, 'r');

   if pointsFileId == -1
      printf("Can't open file \"%s\"\n", pointsFilePath);
      exit(ExitCodes.IOException, "force")
   endif

   rowsCount = 2;
   n = fscanf(pointsFileId, '%d', 1);

   points = fscanf(pointsFileId, '%f', [rowsCount, n]);

   fclose(pointsFileId);

#        READ TRIANGLES
#______________________________

   triangulationFileId = fopen(triangulationFilePath, 'r');

   if triangulationFileId == -1
      printf("Can't open file \"%s\"\n", triangulationFilePath);
      exit(ExitCodes.IOException, "force")
   endif

   rowsCount = 3;

   triangles = fscanf(triangulationFileId, '%d', [rowsCount, Inf])';

   fclose(triangulationFileId);

#      PLOT TRIANGULATION
#______________________________

   figureId = figure('Name', 'Triangulator', 'NumberTitle', 'off');

   triplot(triangles, points(1, :), points(2, :), 'LineStyle', '-',
      'LineWidth', 2, 'Color', 'g', 'Marker', 'o', 'MarkerSize', 2,
      'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');

   grid on;
   axis equal;
   set(gca, 'FontSize', 25);
   title('Delaunay triangulation');

   waitfor(figureId);

   exit(ExitCodes.Ok, "force");
endfunction


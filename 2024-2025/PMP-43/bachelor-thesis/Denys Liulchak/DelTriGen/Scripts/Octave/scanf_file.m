function [ value, count ]= scanf_file(fid, template, sz)
    expected_numel            = prod(sz);
    [ value, count, err_msg ] = fscanf(fid, template, sz);

    if (isfinite(expected_numel) && ...
        expected_numel ~= count)

        if (~isempty(err_msg))
            reason = sprintf('expected %d values, but %d were read; %s', ...
                             expected_numel, count, err_msg);
        else
            reason = sprintf('expected %d values, but %d were read', ...
                             expected_numel, count);
        end

        error('%s: failed to read data from file (%s)', mfilename(), reason);
    end

    if (~isempty(err_msg) && ...
        fgets(fid, 1) ~= -1)
        fseek(fid, -1, 'cof');
        error('%s: failed to read data from file (%s)', mfilename(), err_msg);
    end
end

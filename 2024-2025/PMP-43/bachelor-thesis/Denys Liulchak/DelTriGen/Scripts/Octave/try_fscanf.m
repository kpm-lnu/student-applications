function [ value, count ]= try_fscanf(fid, template, sz)
    try
        [ value, count ] = scanf_file(fid, template, sz);
    catch err
        close_file(fid);
        rethrow(err);
    end
end

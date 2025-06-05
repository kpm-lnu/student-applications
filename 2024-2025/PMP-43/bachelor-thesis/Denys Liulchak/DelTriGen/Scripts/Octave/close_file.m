function close_file(fid)
    status = fclose(fid);

    if (status == -1)
        warning('%s: failed to close the file', mfilename());
    end
end

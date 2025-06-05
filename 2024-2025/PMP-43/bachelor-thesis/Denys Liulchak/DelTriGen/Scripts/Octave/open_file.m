function fid= open_file(fpath, varargin)
    [ fid, err_msg ] = fopen(fpath, varargin{:});

    if (fid == -1)
        error('%s: failed to open the file "%s" (%s)', mfilename(), fpath, err_msg);
    end
end

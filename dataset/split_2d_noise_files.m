clear all
close all

data_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data');
files = {'test_2d_gb1_0902', 'test_2d_titr31'};

for i = 1:numel(files)
    base_name = files{i};
    source_path = fullfile(data_dir, [base_name '.mat']);
    S = load(source_path);

    for noise_idx = 1:3
        real_FFT = S.real_FFT;
        noise_var = sprintf('real_FFTN%d', noise_idx);
        eval([noise_var ' = S.' noise_var ';']);

        noise_level = noise_idx;
        source_file = [base_name '.mat'];
        output_path = fullfile(data_dir, sprintf('%s_noise%d.mat', base_name, noise_idx));

        save(output_path, 'real_FFT', noise_var, 'noise_level', 'source_file', '-v7');
        eval(['clear ' noise_var]);
    end
end

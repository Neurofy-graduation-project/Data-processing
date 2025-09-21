% Improved EEG Wavelet Transform Analysis
curr_dir = pwd;

% the cleaned directory
cleaned_dir = fullfile(curr_dir, 'cleaned');

% Check 
if ~exist(cleaned_dir, 'dir')
    error('Cleaned directory does not exist: %s', cleaned_dir);
end

cleaned_files = dir(fullfile(cleaned_dir, '*.txt'));

if isempty(cleaned_files)
    error('No .txt files found in cleaned directory');
end

fprintf('Found %d files to process\n', length(cleaned_files));


all_conv_3d = {};

% using mexican hat
function psi = mexican_hat_wavelet(t, sigma)
    psi = (2 / (sqrt(3 * sigma) * (pi^0.25))) * ...
          (1 - (t.^2 / sigma^2)) .* exp(-(t.^2 / (2 * sigma^2)));
end

tensor_dir = fullfile(curr_dir, 'tensors');
if ~exist(tensor_dir, 'dir')
    mkdir(tensor_dir);
end

scales = logspace(0, log10(50), 30); 
sampling_rate = 250;
frequencies = sampling_rate ./ (2 * pi * scales); 

for i = 1:length(cleaned_files)
    fprintf('Processing file %d/%d: %s\n', i, length(cleaned_files), cleaned_files(i).name);
    
    % Load the cleaned EEG data
    try
        cleaned_eeg = load(fullfile(cleaned_dir, cleaned_files(i).name));
    catch ME
        warning('Could not load file %s: %s', cleaned_files(i).name, ME.message);
        continue;
    end
    
    % Handle different data structures
    if isstruct(cleaned_eeg)
        fields = fieldnames(cleaned_eeg);
        cleaned_eeg = cleaned_eeg.(fields{1}); % Take first field
    end
    
    [n_samples, channels] = size(cleaned_eeg);
    fprintf('  Data dimensions: %d samples × %d channels\n', n_samples, channels);
    
    % Prepare time vector
    time = (1:n_samples)'; % Column vector
    
    % Initialize 3D tensor for convolution results
    conv_3d = zeros(n_samples, length(scales), channels);
    
    % Perform convolution for each channel
    for ch = 1:channels
        if mod(ch, 10) == 0
            fprintf('    Processing channel %d/%d\n', ch, channels);
        end
        
        % Channel-specific signal
        signal = cleaned_eeg(:, ch);
        
        % Remove DC component and normalize
        signal = detrend(signal);
        signal = signal / std(signal);
        
        % Compute wavelet transform for each scale
        for j = 1:length(scales)
            sigma = scales(j);
            
          
            wavelet_length = min(2 * ceil(6 * sigma), n_samples);
            t_wav = linspace(-3*sigma, 3*sigma, wavelet_length);
            
            
            wavelet = mexican_hat_wavelet(t_wav, sigma);
            
            
            conv_result = conv(signal, wavelet, 'same');
            
            
            conv_3d(:, j, ch) = abs(conv_result);
        end
        
       
        for j = 1:length(scales)
            scale_data = conv_3d(:, j, ch);
            if max(scale_data) > 0
                conv_3d(:, j, ch) = scale_data / max(scale_data);
            end
        end
    end
    
   
    all_conv_3d{i} = conv_3d;
    
    
    tensor_filename = fullfile(tensor_dir, sprintf('eeg_tensor_%03d.h5', i));
    
   
    if exist(tensor_filename, 'file')
        delete(tensor_filename);
    end
    
    try
        % Create and write the main dataset
        h5create(tensor_filename, '/conv_3d', size(conv_3d), 'Datatype', 'single');
        h5write(tensor_filename, '/conv_3d', single(conv_3d));
        
        % Add comprehensive metadata
        h5writeatt(tensor_filename, '/conv_3d', 'channels', channels);
        h5writeatt(tensor_filename, '/conv_3d', 'n_scales', length(scales));
        h5writeatt(tensor_filename, '/conv_3d', 'time_points', n_samples);
        h5writeatt(tensor_filename, '/conv_3d', 'file_source', cleaned_files(i).name);
        h5writeatt(tensor_filename, '/conv_3d', 'sampling_rate', sampling_rate);
        
        % Store scales and corresponding frequencies
        h5create(tensor_filename, '/scales', length(scales));
        h5write(tensor_filename, '/scales', scales);
        
        h5create(tensor_filename, '/frequencies', length(frequencies));
        h5write(tensor_filename, '/frequencies', frequencies);
        
        % Store processing parameters
        h5writeatt(tensor_filename, '/', 'wavelet_type', 'mexican_hat');
        h5writeatt(tensor_filename, '/', 'normalization', 'per_scale_per_channel');
        h5writeatt(tensor_filename, '/', 'processing_date', datestr(now));
        
        fprintf('  Saved tensor to: %s\n', tensor_filename);
        
    catch ME
        warning('Could not save HDF5 file %s: %s', tensor_filename, ME.message);
    end
end

% Save comprehensive metadata file
meta_filename = fullfile(tensor_dir, 'tensors_metadata.h5');
if exist(meta_filename, 'file')
    delete(meta_filename);
end

try
    % Create metadata about the collection
    h5create(meta_filename, '/tensor_count', [1 1]);
    h5write(meta_filename, '/tensor_count', length(cleaned_files));
    
    % Store processing parameters
    h5create(meta_filename, '/scales_used', length(scales));
    h5write(meta_filename, '/scales_used', scales);
    
    h5create(meta_filename, '/frequencies', length(frequencies));
    h5write(meta_filename, '/frequencies', frequencies);
    
    % Add global attributes
    h5writeatt(meta_filename, '/', 'file_count', length(cleaned_files));
    h5writeatt(meta_filename, '/', 'sampling_rate', sampling_rate);
    h5writeatt(meta_filename, '/', 'wavelet_type', 'mexican_hat');
    h5writeatt(meta_filename, '/', 'date_processed', datestr(now));
    h5writeatt(meta_filename, '/', 'matlab_version', version);
    
    fprintf('Saved metadata to: %s\n', meta_filename);
    
catch ME
    warning('Could not save metadata file: %s', ME.message);
end

% Save all tensors together in MATLAB format for backup
try
    backup_filename = fullfile(tensor_dir, 'all_eeg_3d_tensors.mat');
    save(backup_filename, 'all_conv_3d', 'scales', 'frequencies', 'sampling_rate', '-v7.3');
    fprintf('Saved backup MAT file: %s\n', backup_filename);
catch ME
    warning('Could not save backup MAT file: %s', ME.message);
end

fprintf('\nProcessing complete!\n');
fprintf('Processed %d files successfully\n', length(all_conv_3d));
fprintf('Results saved in: %s\n', tensor_dir);
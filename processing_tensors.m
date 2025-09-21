% Get the current working directory
curr_dir = pwd;

% Specify the 'cleaned' directory
cleaned_dir = fullfile(curr_dir,'cleaned');

% Get a list of all .txt files in the 'cleaned' directory
cleaned_files = dir(fullfile(cleaned_dir, '*.txt'));

% Initialize an empty cell array to store the 3D tensors
all_conv_3d = {};

% Define Mexican Hat Wavelet function
function psi = mexican_hat_wavelet(t, width)
    % Mexican Hat Wavelet (Normalized)
    psi = (2 / (sqrt(3 * width) * (pi^0.25))) * ...
          (1 - (t.^2 / width^2)) .* exp(-(t.^2 / (2 * width^2)));
end

% Create a directory for storing tensors if it doesn't exist
tensor_dir = fullfile(curr_dir, 'tensors');
if ~exist(tensor_dir, 'dir')
    mkdir(tensor_dir);
end

% Process each cleaned file
for i = 1:length(cleaned_files)
    % Load the cleaned EEG data
    cleaned_eeg = load(fullfile(cleaned_dir, cleaned_files(i).name));
    channels = size(cleaned_eeg, 2);
    
    % Prepare for wavelet transform
    time = 1:size(cleaned_eeg, 1);
    scales = 1:50; % Adjust scale range as needed
    
    % Initialize 3D tensor for convolution results
    conv_3d = zeros(length(time), length(scales), channels);
    
    % Perform convolution for each channel
    for ch = 1:channels
        % Channel-specific signal
        signal = cleaned_eeg(:, ch);
        
        % Compute wavelet transform
        wav_transform = zeros(length(time), length(scales));
        for j = 1:length(scales)
            width = scales(j);
            
            % Create scaled wavelet
            t = linspace(-width, width, 2*width+1);
            wavelet = mexican_hat_wavelet(t, width);
            
            % Convolution
            conv_result = conv(signal, wavelet, 'same');
            wav_transform(:, j) = abs(conv_result);
        end
        
        % Normalize the transform
        wav_transform = wav_transform / max(wav_transform(:));
        
        % Store in 3D tensor
        conv_3d(:, :, ch) = wav_transform;
    end
    
    % Add the 3D tensor to the cell array
    all_conv_3d{i} = conv_3d;
    
    % Save tensor using HDF5
    tensor_filename = fullfile(tensor_dir, sprintf('eeg_tensor_%d.h5', i));
    
    % Delete the file if it already exists (h5create will error if it exists)
    if exist(tensor_filename, 'file')
        delete(tensor_filename);
    end
    
    % Create and write the dataset
    h5create(tensor_filename, '/conv_3d', size(conv_3d));
    h5write(tensor_filename, '/conv_3d', conv_3d);
    
    % Add metadata attributes
    h5writeatt(tensor_filename, '/conv_3d', 'channels', channels);
    h5writeatt(tensor_filename, '/conv_3d', 'scales', length(scales));
    h5writeatt(tensor_filename, '/conv_3d', 'time_points', length(time));
    h5writeatt(tensor_filename, '/conv_3d', 'file_source', cleaned_files(i).name);
    
    % Also store the scales for reference
    h5create(tensor_filename, '/scales', length(scales));
    h5write(tensor_filename, '/scales', scales);
    
    disp(['Saved tensor ' num2str(i) ' to HDF5 file']);
end

% Save a metadata file with information about all tensors
meta_filename = fullfile(tensor_dir, 'tensors_metadata.h5');
if exist(meta_filename, 'file')
    delete(meta_filename);
end

% Create metadata about the collection
h5create(meta_filename, '/tensor_count', [1 1]);
h5write(meta_filename, '/tensor_count', length(cleaned_files));

% Create a list of filenames
filenames = cell(length(cleaned_files), 1);
for i = 1:length(cleaned_files)
    filenames{i} = cleaned_files(i).name;
end

% Save filenames as a string array attribute
h5writeatt(meta_filename, '/', 'file_count', length(cleaned_files));
h5writeatt(meta_filename, '/', 'date_processed', datestr(now));

% Also save all tensors together in MATLAB format for backup
save(fullfile(tensor_dir, 'all_eeg_3d_tensors.mat'), 'all_conv_3d', '-v7.3');
disp('Processing complete');
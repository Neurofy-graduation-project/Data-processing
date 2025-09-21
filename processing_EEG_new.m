function batchCleaning()
    
    current_dir = pwd;
    fprintf('Current working directory: %s\n', current_dir);
    
   % the case --> it varies
    case_dir = fullfile(current_dir, 'ch02\');
    fprintf('Looking for EDF files in: %s\n', case_dir);
    
    
    if ~exist(case_dir, 'dir')
        error('Input directory does not exist: %s', case_dir);
    end
    
    % Get list of all EDF files in the directory
    edf_files = dir(fullfile(case_dir, 'chb02_*.edf'));
    if isempty(edf_files)
        error('No EDF files found in directory: %s', case_dir);
    end
    fprintf('Found %d EDF files\n', length(edf_files));
    
    % Create for cleaned data
    cleaned_dir = fullfile(case_dir, 'cleaned');
    if ~exist(cleaned_dir, 'dir')
        [success, msg] = mkdir(cleaned_dir);
        if ~success
            error('Failed to create output directory: %s\nError: %s', cleaned_dir, msg);
        end
    end
    fprintf('Output directory: %s\n', cleaned_dir);
    
    
    test_file = fullfile(cleaned_dir, 'test_write.txt');
    try
        fid = fopen(test_file, 'w');
        if fid == -1
            error('Cannot write to output directory: %s', cleaned_dir);
        end
        fclose(fid);
        delete(test_file);
        fprintf('Successfully verified write permissions\n');
    catch ME
        error('Write permission test failed: %s', ME.message);
    end
    
    
    for i = 1:length(edf_files)
        try
            current_file = fullfile(case_dir, edf_files(i).name);
            fprintf('\nProcessing file %d/%d: %s\n', i, length(edf_files), edf_files(i).name);
            
            
            [sig, header] = edfread(current_file);
            
             
            data = table2array(sig);
            
            % Get the FP2-F8 channel
            channel_names = sig.Properties.VariableNames;
            fp2f8_idx = find(contains(channel_names, 'FP2_F8') | ...
                           contains(channel_names, 'SignalLabel13_FP2_F8'));
            
            if isempty(fp2f8_idx)
                warning('FP2-F8 channel not found in %s. Skipping file.', edf_files(i).name);
                continue;
            end
            
           
            fp2f8_cell = data(:, fp2f8_idx);
            fp2f8_data = cell2mat(fp2f8_cell);
            fp2f8_data = fp2f8_data(:);
            
            % Get sampling rate
            if isfield(header, 'samplingrate')
                fs = header.samplingrate(1);
            elseif isfield(header, 'samples')
                fs = header.samples(1) / header.duration;
            else
                warning('Sampling rate not found in header. Using default 256 Hz');
                fs = 256;
            end
            
            
            fp2f8_data = preprocess_eeg(fp2f8_data, fs);
            
            
            clean_eeg = process_with_overlapping_windows(fp2f8_data, fs);
            
            
            clean_eeg = postprocess_eeg(clean_eeg, fs);
            
            
            [~, name, ~] = fileparts(edf_files(i).name);
            output_file = fullfile(cleaned_dir, [name '_cleaned.txt']);
            
           
            try
                % Calculate quality metrics
                snr_original = calculate_snr(fp2f8_data);
                snr_cleaned = calculate_snr(clean_eeg);
                artifact_reduction = calculate_artifact_reduction(fp2f8_data, clean_eeg);
                
                % Save cleaned signal
                writematrix(clean_eeg, output_file);
                fprintf('Saved cleaned signal to: %s\n', output_file);
                
                % Save quality metrics
                metrics_file = fullfile(cleaned_dir, [name '_metrics.txt']);
                fid = fopen(metrics_file, 'w');
                fprintf(fid, 'Signal Quality Metrics\n');
                fprintf(fid, '=====================\n');
                fprintf(fid, 'Original SNR: %.2f dB\n', snr_original);
                fprintf(fid, 'Cleaned SNR: %.2f dB\n', snr_cleaned);
                fprintf(fid, 'SNR Improvement: %.2f dB\n', snr_cleaned - snr_original);
                fprintf(fid, 'Artifact Reduction: %.2f%%\n', artifact_reduction * 100);
                fclose(fid);
                
                fprintf('SNR improvement: %.2f dB, Artifact reduction: %.1f%%\n', ...
                    snr_cleaned - snr_original, artifact_reduction * 100);
                
            catch ME
                error('Failed to save results: %s', ME.message);
            end
            
            % Create enhanced visualization
            try
                create_enhanced_visualization(fp2f8_data, clean_eeg, fs, name, cleaned_dir);
            catch ME
                warning('Failed to create visualization: %s', ME.message);
            end
            
        catch ME
            warning('Error processing file %s: %s', edf_files(i).name, ME.message);
            continue;
        end
    end
    fprintf('\nProcessing complete! Cleaned files are saved in: %s\n', cleaned_dir);
end

function clean_data = process_with_overlapping_windows(data, fs)
    % Enhanced windowing approach with overlap-add reconstruction
    
    % Window parameters (optimized for EEG)
    window_duration = 4;  % 4 seconds
    overlap_percent = 50; % 50% overlap
    
    window_samples = round(window_duration * fs);
    overlap_samples = round(window_samples * overlap_percent / 100);
    hop_size = window_samples - overlap_samples;
    
    % Initialize output
    clean_data = zeros(size(data));
    weight_sum = zeros(size(data));
    
    % Windowing function for smooth transitions
    window_func = hann(window_samples);
    
    fprintf('Processing with %d-second windows, %d%% overlap\n', window_duration, overlap_percent);
    
    % Process overlapping windows
    num_windows = ceil((length(data) - overlap_samples) / hop_size);
    fprintf('Total windows to process: %d\n', num_windows);
    
    for win_idx = 1:num_windows
        % Calculate window boundaries
        start_idx = (win_idx - 1) * hop_size + 1;
        end_idx = min(start_idx + window_samples - 1, length(data));
        current_length = end_idx - start_idx + 1;
        
        if current_length < window_samples / 2  % Skip very short windows
            continue;
        end
        
        % Extract current window
        current_window = data(start_idx:end_idx);
        
        % Pad if necessary
        if current_length < window_samples
            padded_window = [current_window; zeros(window_samples - current_length, 1)];
            current_window_func = window_func(1:current_length);
        else
            padded_window = current_window;
            current_window_func = window_func;
        end
        
        % Apply artifact removal
        clean_window = enhanced_sart(padded_window, fs);
        
        % Trim back to original length if padded
        if current_length < window_samples
            clean_window = clean_window(1:current_length);
        end
        
        % Apply windowing function for smooth blending
        clean_window = clean_window .* current_window_func;
        
        % Add to output with overlap-add
        clean_data(start_idx:end_idx) = clean_data(start_idx:end_idx) + clean_window;
        weight_sum(start_idx:end_idx) = weight_sum(start_idx:end_idx) + current_window_func;
        
        % Progress indicator
        if mod(win_idx, 10) == 0 || win_idx == num_windows
            fprintf('Processed window %d/%d (%.1f%%)\n', win_idx, num_windows, 100*win_idx/num_windows);
        end
    end
    
    % Normalize by window overlap
    clean_data(weight_sum > 0) = clean_data(weight_sum > 0) ./ weight_sum(weight_sum > 0);
end

function clean_data = enhanced_sart(contaminated_data, fs)
    % Enhanced SART with adaptive parameters
    n = length(contaminated_data);
    
    % Adaptive parameters based on signal characteristics
    signal_power = var(contaminated_data);
    noise_estimate = estimate_noise_level(contaminated_data);
    
    num_iterations = min(100, max(20, round(50 * noise_estimate / signal_power)));
    lambda = 0.01 * (1 + noise_estimate / signal_power);
    
    % Initialize
    clean_data = contaminated_data;
    
    % Iterative cleaning with convergence check
    prev_residual = inf;
    for iter = 1:num_iterations
        % Compute residual
        residual = contaminated_data - clean_data;
        residual_norm = norm(residual);
        
        % Check for convergence
        if abs(prev_residual - residual_norm) / prev_residual < 1e-6
            fprintf('  Converged at iteration %d\n', iter);
            break;
        end
        prev_residual = residual_norm;
        
        % Update estimate
        clean_data = clean_data + lambda * residual;
        
        % Adaptive soft thresholding
        threshold = estimate_threshold(clean_data, fs);
        clean_data = soft_threshold(clean_data, threshold);
        
        % Reduce lambda over iterations for stability
        lambda = lambda * 0.99;
    end
end

function threshold = estimate_threshold(data, fs)
    % Adaptive threshold based on signal characteristics
    n = length(data);
    sigma = mad(data, 1) / 0.6745;  % Robust noise estimate
    threshold = sigma * sqrt(2 * log(n));
    
    % Adjust based on frequency content
    [pxx, f] = periodogram(data, [], [], fs);
    
    % Higher threshold if high-frequency content dominates
    high_freq_power = sum(pxx(f > fs/4)) / sum(pxx);
    if high_freq_power > 0.3
        threshold = threshold * 1.5;
    end
end

function data_thresh = soft_threshold(data, threshold)
    % Soft thresholding function
    data_thresh = sign(data) .* max(abs(data) - threshold, 0);
end

function noise_level = estimate_noise_level(data)
    % Estimate noise level using robust statistics
    % Use median absolute deviation in high-frequency content
    diff_data = diff(data);
    noise_level = mad(diff_data, 1) / 0.6745;
end

function data = preprocess_eeg(data, fs)
    % Enhanced preprocessing --> additional%%%%%
    % Remove DC offset
    data = detrend(data, 'constant');
    
    % Remove linear trend
    data = detrend(data, 'linear');
    
    % Design filters with improved specifications
    nyquist = fs/2;
    
    % Notch filters 
    for freq = [50, 60, 100, 120]
        if freq < nyquist
            wo = freq/nyquist;
            bw = wo/50;  % Narrower bandwidth
            [b,a] = iirnotch(wo, bw);
            data = filtfilt(b, a, data);
        end
    end
    
    % Improved bandpass filter 
    low_cutoff = 0.5;
    high_cutoff = min(45, nyquist * 0.8);
    
    [b,a] = butter(6, [low_cutoff high_cutoff]/nyquist, 'bandpass');
    data = filtfilt(b, a, data);
end

function data = postprocess_eeg(data, fs)
    % post-processing
    % DC offset
    data = detrend(data, 'constant');
    
    % median filtering 
    window_size = round(fs/50);  % 20ms window
    if mod(window_size,2) == 0
        window_size = window_size + 1;
    end
    data = medfilt1(data, min(window_size, 15));  
    
    % Final smoothing
    nyquist = fs/2;
    cutoff = min(40, nyquist * 0.7);
    [b,a] = butter(4, cutoff/nyquist, 'low');
    data = filtfilt(b, a, data);
end

function snr_db = calculate_snr(signal)
    % SNR 
    signal_power = var(signal);
    noise_power = var(diff(signal)) / 2;  % Estimate noise
    snr_db = 10 * log10(signal_power / noise_power);
end

function reduction = calculate_artifact_reduction(original, cleaned)
    % artifact reduction
    original_artifacts = detect_artifacts(original);
    cleaned_artifacts = detect_artifacts(cleaned);
    
    if original_artifacts == 0
        reduction = 0;
    else
        reduction = (original_artifacts - cleaned_artifacts) / original_artifacts;
    end
    reduction = max(0, reduction);  % Ensure non-negative
end

function artifact_count = detect_artifacts(data)
    % Simple artifact detection based on amplitude thresholds
    threshold = 5 * std(data);
    artifact_count = sum(abs(data) > threshold);
end

function create_enhanced_visualization(original, cleaned, fs, name, output_dir)

    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
    

    t = (0:length(original)-1) / fs;
    
    
    subplot(2,2,1);
    plot(t, original, 'b-', 'LineWidth', 0.5);
    hold on;
    plot(t, cleaned, 'r-', 'LineWidth', 0.5);
    title(['Time Domain: ' strrep(name, '_', '\_')]);
    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    legend('Original', 'Cleaned', 'Location', 'best');
    grid on;
    

    subplot(2,2,2);
    [pxx_orig, f] = periodogram(original, [], [], fs);
    [pxx_clean, ~] = periodogram(cleaned, [], [], fs);
    
    semilogy(f, pxx_orig, 'b-', 'LineWidth', 1);
    hold on;
    semilogy(f, pxx_clean, 'r-', 'LineWidth', 1);
    title('Power Spectral Density');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (µV²/Hz)');
    legend('Original', 'Cleaned', 'Location', 'best');
    grid on;
    xlim([0, min(fs/2, 50)]);
    
    % Difference signal
    subplot(2,2,3);
    difference = original - cleaned;
    plot(t, difference, 'g-', 'LineWidth', 0.5);
    title('Removed Artifacts (Original - Cleaned)');
    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    grid on;
    
   
    subplot(2,2,4);
    histogram(original, 50, 'Normalization', 'probability', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    hold on;
    histogram(cleaned, 50, 'Normalization', 'probability', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    title('Amplitude Distribution');
    xlabel('Amplitude (µV)');
    ylabel('Probability');
    legend('Original', 'Cleaned', 'Location', 'best');
    grid on;
    
  
    plot_file = fullfile(output_dir, [name '_enhanced_plot.png']);
    saveas(fig, plot_file, 'png');
    close(fig);
    
    fprintf('Saved enhanced plot to: %s\n', plot_file);
end
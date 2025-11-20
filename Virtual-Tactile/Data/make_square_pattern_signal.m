function [signal, t, pos] = make_square_pattern_signal(width, space_ratio, height, noise_level)
    % width: grating 너비 (um)
    % space_ratio: grating 간 거리 비율
    % height: square wave 높이
    % noise_level: 노이즈 세기 (height 대비 비율, 예: 0.05)
    
    total_length = 50000;     % 전체 이동 거리 (um, 5cm)
    velocity = 20000;         % 이동 속도 (um/s, 2cm/s)
    fs = 2500;                % 샘플링 레이트 (Hz)
    total_time = 300;         % 측정 시간 (초)
    N = fs * total_time;      % 전체 샘플 개수

    t = linspace(0, total_time, N);   % 시간 벡터
    pos = velocity * t;               % 누적 이동 거리 (um)
    
    period = width + width * space_ratio;   % 한 주기 길이 (um)
    phase = mod(pos, period);

    signal = zeros(1, N);

    % 위쪽 피크(높이 구간) 인덱스만 별도 마스킹
    idx_peak = phase < width;
    % 위쪽 피크 구간에만 height와 노이즈 적용
    signal(idx_peak) = height + noise_level * height * randn(1, sum(idx_peak));

    % 아래쪽(0 구간)은 이미 0이므로 따로 노이즈 추가하지 않음

    % 필요 시 시각화
    plot(t, signal); xlabel('Time (s)'); ylabel('Signal'); title('Expected Sensor Signal');, xlim([0 3])
end
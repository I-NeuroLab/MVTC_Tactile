function [signal, t, pos, segments] = make_square_pattern_signal_scanpause_segment(width, space_ratio, height, noise_level)
    % 가로 패턴 square wave 센서 신호 시뮬레이션 + 세그먼트 분할 (오류 안전 처리)
    % 각 cycle: 1초 대기, 2.5초 이동, 1초 대기 (총 4.5초)
    % 각 segment: 2.5초 이동 앞뒤 0.25초씩 포함 (총 3초)
    % 
    % 입력: width         - grating 너비 (um)
    %      space_ratio   - grating 간 거리 비율
    %      height        - square wave 높이
    %      noise_level   - 노이즈 세기 (height 대비 비율)
    % 출력: signal       - 전체 시뮬레이션 신호
    %        t           - 시간 벡터
    %        pos         - 이동 거리 벡터
    %        segments    - 각 세그먼트 (행: 세그먼트, 열: 샘플)

    % ----- 파라미터 -----
    total_time = 300;     % 전체 측정 시간 (초, 5분)
    fs = 2500;            % 샘플링 레이트 (Hz)
    N = fs * total_time;  % 전체 샘플 개수

    % cycle 조건
    pause_pre = 1.0;      % 이동 전 대기 (초)
    move = 2.5;           % 이동 (초)
    pause_post = 1.0;     % 이동 후 대기 (초)
    cycle = pause_pre + move + pause_post;   % 4.5초
    cycle_len = round(cycle * fs);

    % ----- 한 cycle 신호 생성 -----
    velocity = 20000;    % 이동 속도 (um/s, 2cm/s)
    t_cycle = linspace(0, cycle, cycle_len);
    signal_cycle = zeros(1, cycle_len);
    pos_cycle = zeros(1, cycle_len);

    % 이동구간 인덱스
    idx_move = round(pause_pre*fs)+1 : round((pause_pre+move)*fs);
    t_move = t_cycle(idx_move) - t_cycle(idx_move(1));  % 0 ~ move초
    pos_move = velocity * t_move;

    period = width + width * space_ratio;   % 한 주기 길이 (um)
    phase = mod(pos_move, period);
    sig_move = zeros(1, length(idx_move));
    idx_peak = phase < width;
    sig_move(idx_peak) = height + noise_level * height * randn(1, sum(idx_peak));
    % 아래쪽(space, 0 구간)은 노이즈 없음

    signal_cycle(idx_move) = sig_move;
    pos_cycle(idx_move) = pos_move;

    % ----- 전체 신호 반복 생성 및 길이 맞추기 -----
    n_cycles = floor(N / cycle_len);
    signal = repmat(signal_cycle, 1, n_cycles);
    pos = repmat(pos_cycle, 1, n_cycles);
    t = linspace(0, cycle*n_cycles, cycle_len*n_cycles);

    % 길이 맞춤(패딩 또는 잘라내기)
    siglen = length(signal);
    if siglen < N
        signal = [signal, zeros(1, N-siglen)];
        pos = [pos, zeros(1, N-siglen)];
        t = [t, zeros(1, N-siglen)];
    else
        signal = signal(1:N);
        pos = pos(1:N);
        t = t(1:N);
    end

    % ----- segmentation: 이동 2.5초 앞뒤 0.25초 포함(총 3초) -----
    seg_pre = 0.25;   % 앞쪽(대기 일부) (초)
    seg_move = 2.5;   % 이동구간 (초)
    seg_post = 0.25;  % 뒤쪽(대기 일부) (초)
    seg_len = round((seg_pre + seg_move + seg_post) * fs);   % 3초 구간 샘플 수

    segments_cell = {};
    seg_time = linspace(-seg_pre, seg_move+seg_post, seg_len); 
    i_seg = 1;
    for i = 1:n_cycles
        base_idx = (i-1)*cycle_len + 1;
        seg_start = round((pause_pre-seg_pre)*fs) + base_idx;
        seg_end = seg_start + seg_len - 1;
        % 경계 내부 + 정확한 길이만 저장
        if seg_start > 0 && seg_end <= N && (seg_end-seg_start+1)==seg_len
            segments_cell{i_seg,1} = signal(seg_start:seg_end);
            i_seg = i_seg + 1;
        end
    end
    if isempty(segments_cell)
        segments = [];
    else
        segments = cell2mat(segments_cell);
    end

    % ----- 첫/마지막 segment plot -----
    if ~isempty(segments)
        figure(1);
        % subplot(2,1,1);
        plot(seg_time, segments(1,:));
        xlabel('Time (s)');
        ylabel('Signal');
        title('First segment (3초: 0.25초+2.5초+0.25초)');

        % subplot(2,1,2);
        % plot(seg_time, segments(end,:));
        % xlabel('Time (s)');
        % ylabel('Signal');
        % title('Last segment (3초: 0.25초+2.5초+0.25초)');
    end
end

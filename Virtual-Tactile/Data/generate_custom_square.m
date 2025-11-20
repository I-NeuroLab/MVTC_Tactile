function wave = generate_custom_square(width, space_ratio, height)
    % width: 그레이팅 한 주기의 길이(샘플 단위)
    % space_ratio: 그레이팅 사이 거리 / width (예: 1이면 width만큼 간격)
    % height: square wave의 최고-최저 차이
    % 출력: wave (2000Hz, 6분, 720,000 샘플)
    
    fs = 2500; % 샘플링 레이트 (Hz)
    duration = 300; % 6분 = 360초
    total_samples = fs * duration; % 전체 샘플 개수
    
    % 단일 패턴(그레이팅 + 스페이스) 만들기
    grating = ones(1, width) * (height/2);           % 그레이팅 구간 (high)
    space = ones(1, round(width * space_ratio)) * (-height/2);  % 스페이스 구간 (low)
    pattern = [grating, space];
    
    % 패턴 반복하여 전체 wave 생성
    n_repeat = ceil(total_samples / length(pattern));
    wave = repmat(pattern, 1, n_repeat);
    wave = wave(1:total_samples)+(height/2); % 딱 맞게 자르기
end
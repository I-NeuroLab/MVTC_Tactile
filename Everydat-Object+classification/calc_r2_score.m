function val = calc_r2_score(y_true, y_pred)
    y_true = double(y_true(:));
    y_pred = double(y_pred(:));

    ss_res = sum((y_true - y_pred).^2);

    ss_tot = sum((y_true - mean(y_true)).^2);

    if ss_tot == 0
        val = 0;
    else
        val = 1 - ss_res / ss_tot;
        % val = 1 - ((ss_res/(length(y_true)-1-1)) / (ss_tot/(length(y_true))));
    end
end
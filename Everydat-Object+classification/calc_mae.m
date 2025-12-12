function val = calc_mae(y_true, y_pred)
    y_true = double(y_true(:));
    y_pred = double(y_pred(:));

    val = mean(abs(y_true - y_pred));
end
function val = calc_rmse(y_true, y_pred)
    y_true = double(y_true);
    y_pred = double(y_pred);

    val = sqrt(mean((y_true(:) - y_pred(:)).^2));
end
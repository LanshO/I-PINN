clear;
close all;

% Relative paths inside the open-source package
script_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(script_dir, '..');
example_dir = fullfile(repo_root, 'example_case');
result_dir = fullfile(repo_root, 'results');
figure_dir = fullfile(result_dir, 'comparison_figures_matlab');

if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

hole_radius = 40;

pred_u = csvread(fullfile(result_dir, 'displacement_u.csv'))';
pred_v = csvread(fullfile(result_dir, 'displacement_v.csv'))';
pred_exx = csvread(fullfile(result_dir, 'strain_xx.csv'))';
pred_exy = csvread(fullfile(result_dir, 'strain_xy.csv'))';
pred_eyy = csvread(fullfile(result_dir, 'strain_yy.csv'))';
pred_sxx = csvread(fullfile(result_dir, 'stress_xx.csv'))';
pred_sxy = csvread(fullfile(result_dir, 'stress_xy.csv'))';
pred_syy = csvread(fullfile(result_dir, 'stress_yy.csv'))';

if max(abs(pred_sxx(:))) < 1e3
    pred_sxx = pred_sxx * 1e9;
    pred_sxy = pred_sxy * 1e9;
    pred_syy = pred_syy * 1e9;
end

fem_u = csvread(fullfile(example_dir, 'u001.csv'));
fem_v = csvread(fullfile(example_dir, 'v001.csv'));
fem_exx = csvread(fullfile(example_dir, 'strain_xx001.csv'));
fem_exy = csvread(fullfile(example_dir, 'strain_xy001.csv'));
fem_eyy = csvread(fullfile(example_dir, 'strain_yy001.csv'));
fem_sxx = csvread(fullfile(example_dir, 'stress_xx001.csv'));
fem_sxy = csvread(fullfile(example_dir, 'stress_xy001.csv'));
fem_syy = csvread(fullfile(example_dir, 'stress_yy001.csv'));

fem_u = fem_u(21:401, 21:801);
fem_v = fem_v(21:401, 21:801);
fem_exx = fem_exx(21:401, 21:801);
fem_exy = fem_exy(21:401, 21:801);
fem_eyy = fem_eyy(21:401, 21:801);
fem_sxx = fem_sxx(21:401, 21:801);
fem_sxy = fem_sxy(21:401, 21:801);
fem_syy = fem_syy(21:401, 21:801);

% The FEM export stores engineering shear strain gamma_xy, whereas the
% network predicts the tensor shear strain epsilon_xy. Convert gamma_xy
% to epsilon_xy before visualization and error comparison.
fem_exy = fem_exy / 2;

pred_sxx = pred_sxx / 1e6;
pred_sxy = pred_sxy / 1e6;
pred_syy = pred_syy / 1e6;
fem_sxx = fem_sxx / 1e6;
fem_sxy = fem_sxy / 1e6;
fem_syy = fem_syy / 1e6;

[rows, cols] = size(pred_u);
for i = 1:rows
    for j = 1:cols
        if (i - floor(rows / 2 + 1))^2 + (j - floor(cols / 2 + 1))^2 < hole_radius * hole_radius
            pred_u(i, j) = nan; pred_v(i, j) = nan;
            pred_exx(i, j) = nan; pred_exy(i, j) = nan; pred_eyy(i, j) = nan;
            pred_sxx(i, j) = nan; pred_sxy(i, j) = nan; pred_syy(i, j) = nan;
            fem_u(i, j) = nan; fem_v(i, j) = nan;
            fem_exx(i, j) = nan; fem_exy(i, j) = nan; fem_eyy(i, j) = nan;
            fem_sxx(i, j) = nan; fem_sxy(i, j) = nan; fem_syy(i, j) = nan;
        end
    end
end

save_comparison(pred_u, fem_u, 'u', '', fullfile(figure_dir, 'compare_u.png'));
save_comparison(pred_v, fem_v, 'v', '', fullfile(figure_dir, 'compare_v.png'));
save_comparison(pred_exx, fem_exx, '\epsilon_{xx}', '', fullfile(figure_dir, 'compare_exx.png'));
save_comparison(pred_exy, fem_exy, '\epsilon_{xy}', '', fullfile(figure_dir, 'compare_exy.png'));
save_comparison(pred_eyy, fem_eyy, '\epsilon_{yy}', '', fullfile(figure_dir, 'compare_eyy.png'));
save_comparison(pred_sxx, fem_sxx, '\sigma_{xx}', 'MPa', fullfile(figure_dir, 'compare_sxx.png'));
save_comparison(pred_sxy, fem_sxy, '\sigma_{xy}', 'MPa', fullfile(figure_dir, 'compare_sxy.png'));
save_comparison(pred_syy, fem_syy, '\sigma_{yy}', 'MPa', fullfile(figure_dir, 'compare_syy.png'));

function save_comparison(pred_data, fem_data, field_label, unit_label, save_path)
    err_data = pred_data - fem_data;
    pair_min = min([min(pred_data(:), [], 'omitnan'), min(fem_data(:), [], 'omitnan')]);
    pair_max = max([max(pred_data(:), [], 'omitnan'), max(fem_data(:), [], 'omitnan')]);
    finite_err = abs(err_data(~isnan(err_data)));
    err_lim = quantile(finite_err, 0.995);
    if err_lim <= 0
        err_lim = max(finite_err);
    end

    fig = figure('Visible', 'off', 'Position', [100 100 1500 420]);
    tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    h = imagesc(pred_data);
    set(h, 'alphadata', ~isnan(pred_data));
    axis image off;
    colormap(gca, jet);
    caxis([pair_min, pair_max]);
    c = colorbar;
    if ~isempty(unit_label)
        ylabel(c, unit_label);
    end
    title(['I-PINN result of ', field_label], 'Interpreter', 'tex');

    nexttile;
    h = imagesc(fem_data);
    set(h, 'alphadata', ~isnan(fem_data));
    axis image off;
    colormap(gca, jet);
    caxis([pair_min, pair_max]);
    c = colorbar;
    if ~isempty(unit_label)
        ylabel(c, unit_label);
    end
    title(['FEM result of ', field_label], 'Interpreter', 'tex');

    nexttile;
    h = imagesc(err_data);
    set(h, 'alphadata', ~isnan(err_data));
    axis image off;
    colormap(gca, jet);
    caxis([-err_lim, err_lim]);
    c = colorbar;
    if ~isempty(unit_label)
        ylabel(c, unit_label);
    end
    title(['Difference of ', field_label, ' (I-PINN - FEM)'], 'Interpreter', 'tex');

    exportgraphics(fig, save_path, 'Resolution', 300);
    close(fig);
end

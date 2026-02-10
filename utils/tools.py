import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from einops import rearrange
from tqdm import tqdm
import datetime
from utils.metrics import metric

plt.switch_backend("agg")


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: (
                args.learning_rate
                if epoch < 3
                else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
            )
        }
    elif args.lradj == "type4":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch) // 1))}
    elif args.lradj == "PEMS":
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            if accelerator is not None:
                accelerator.print("Updating learning rate to {}".format(lr))
            else:
                print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(
        self, accelerator=None, patience=7, verbose=True, delta=0, save_mode=True
    ):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode
        self.iteration = 0

    def __call__(self, val_loss, model, path):
        self.iteration += 1
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            else:
                self.accelerator.print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )
            else:
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )

        # Prepare model state dict
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            state_dict = model.state_dict()
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = model.state_dict()
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        save_path = path + "/" + "checkpoint.pth"
        torch.save(state_dict, save_path)
        print(f"Best model saved to: {save_path}")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(
    args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, mape_metric
):
    total_loss = []
    total_mae_loss = []
    total_mape_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(vali_loader)
        ):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(accelerator.device)
            )

            # Forward pass
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    (
                        outputs,
                        alignment_loss,
                        source_embeddings_out,
                        enc_out_patch,
                        enc_out_reprog,
                        gate_weights,
                    ) = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs
                else:
                    (
                        outputs,
                        alignment_loss,
                        source_embeddings_out,
                        enc_out_patch,
                        enc_out_reprog,
                        gate_weights,
                    ) = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]
            batch_y = batch_y[:, -args.pred_len :, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            mape_loss = mape_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            total_mape_loss.append(mape_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    total_mape_loss = np.average(total_mape_loss)

    model.train()
    return total_loss, total_mae_loss, total_mape_loss


def test(
    args,
    accelerator,
    model,
    train_loader,
    vali_loader,
    criterion,
    mae_metric,
    mape_metric,
):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    total_loss = []
    total_mae_loss = []
    total_mape_loss = []
    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len :, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i] : id_list[i + 1], :, :] = model(
                x[id_list[i] : id_list[i + 1]],
                None,
                dec_inp[id_list[i] : id_list[i + 1]],
                None,
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == "MS" else 0
        outputs = outputs[:, -args.pred_len :, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(pred.detach(), true.detach())
        mae_loss = mae_metric(pred.detach(), true.detach())
        mape_loss = mape_metric(pred.detach(), true.detach())
        total_loss.append(loss.item())
        total_mae_loss.append(mae_loss.item())
        total_mape_loss.append(mape_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    total_mape_loss = np.average(total_mape_loss)
    model.train()
    return total_loss, total_mae_loss, total_mape_loss


def load_content(args):
    if "ETT" in args.data:
        file = "ETT"
    else:
        file = args.data
    with open(
        "./dataset/prompt_bank/{0}.txt".format(file),
        "r",
    ) as f:
        content = f.read()
    return content


def visualize_holiday_predictions(
    args, model, test_data, test_loader, accelerator, criterion, mae_metric, mape_metric
):
    """Visualize model predictions during holiday periods."""
    import pandas as pd
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import os
    from datetime import datetime, timedelta

    def extract_central_tendency(data, window_hours=48):
        """Extract central tendency using rolling average to show overall trend."""
        return data.rolling(window=window_hours, center=True, min_periods=1).mean()

    # Create output directory for visualizations
    os.makedirs("holiday_predictions", exist_ok=True)

    # Read holiday data
    holiday_data = pd.read_excel(args.holiday_data_path)
    print("Holiday data loaded:")
    print(holiday_data)

    # Process holiday data to get the holiday periods
    holiday_periods = {}
    for _, row in holiday_data.iterrows():
        holiday_name = row["holiday"]

        for year in ["2017", "2018", "2019"]:
            start_col = f"start_date_{year}"
            end_col = f"end_date_{year}"

            if pd.notna(row[start_col]) and pd.notna(row[end_col]):
                start_date = pd.to_datetime(row[start_col])
                end_date = pd.to_datetime(row[end_col])
                print(
                    f"Added holiday: {holiday_name} ({year}): {start_date} to {end_date}"
                )
                holiday_periods[(start_date, end_date)] = holiday_name

    # Run test to get predictions
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(accelerator.device)
            )

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, _, _, _, _, _ = model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark
                    )
            else:
                outputs, _, _, _, _, _ = model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]
            batch_y = batch_y[:, -args.pred_len :, f_dim:].to(accelerator.device)

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    print(f"Prediction shape: {preds.shape}")

    if len(preds.shape) == 4:
        num_samples = preds.shape[0] * preds.shape[1]
        preds = preds.reshape(num_samples, preds.shape[2], preds.shape[3])
        trues = trues.reshape(num_samples, trues.shape[2], trues.shape[3])

    print(f"Reshaped prediction shape: {preds.shape}")

    # Get the timestamps for test set
    test_start_idx = 20 * 30 * 24
    raw_data = pd.read_csv(os.path.join(args.root_path, args.data_path))
    raw_data["date"] = pd.to_datetime(raw_data["date"])

    if test_start_idx + preds.shape[0] + args.pred_len > len(raw_data):
        print(f"Warning: Not enough dates in the dataset. Adjusting to available data.")
        test_dates = raw_data["date"][test_start_idx:].reset_index(drop=True)
    else:
        test_dates = raw_data["date"][
            test_start_idx : test_start_idx + preds.shape[0] + args.pred_len
        ].reset_index(drop=True)

    print(f"Test dates: {test_dates.iloc[0]} to {test_dates.iloc[-1]}")

    max_window_start = len(test_dates) - args.pred_len

    # Analyze all windows for holiday coverage
    individual_holiday_windows = {}
    all_window_info = {}

    for i in range(min(preds.shape[0], max_window_start)):
        window_dates = test_dates[i : i + args.pred_len].reset_index(drop=True)
        window_start = window_dates.iloc[0]
        window_end = window_dates.iloc[-1]

        window_holidays = []
        for (holiday_start, holiday_end), holiday_name in holiday_periods.items():
            if not ((window_end < holiday_start) or (window_start > holiday_end)):
                holiday_coverage = min(holiday_end, window_end) - max(
                    holiday_start, window_start
                )
                coverage_ratio = holiday_coverage / (holiday_end - holiday_start)

                window_holidays.append(
                    {
                        "holiday_name": holiday_name,
                        "holiday_start": holiday_start,
                        "holiday_end": holiday_end,
                        "coverage_ratio": coverage_ratio,
                    }
                )

        if window_holidays:
            all_window_info[i] = {
                "window_idx": i,
                "window_start": window_start,
                "window_end": window_end,
                "holidays": window_holidays,
                "window_dates": window_dates,
                "holiday_count": len(window_holidays),
            }

    # Select the best window for each individual holiday
    for i, window_info in all_window_info.items():
        for holiday in window_info["holidays"]:
            holiday_name = holiday["holiday_name"]
            holiday_year = holiday["holiday_start"].year
            holiday_key = f"{holiday_name}_{holiday_year}"

            if holiday_key not in individual_holiday_windows:
                individual_holiday_windows[holiday_key] = {
                    "best_window": None,
                    "best_centrality": float("inf"),
                }

            window_start = window_info["window_start"]
            window_end = window_info["window_end"]
            window_length = (window_end - window_start).total_seconds()
            window_center = window_start + pd.Timedelta(seconds=window_length / 2)

            holiday_center = holiday["holiday_start"] + (
                holiday["holiday_end"] - holiday["holiday_start"]
            ) / 2
            centrality = abs((holiday_center - window_center).total_seconds())

            if centrality < individual_holiday_windows[holiday_key]["best_centrality"]:
                individual_holiday_windows[holiday_key]["best_centrality"] = centrality
                individual_holiday_windows[holiday_key]["best_window"] = window_info

    print(
        f"Selected best windows for {len(individual_holiday_windows)} individual holidays"
    )

    # Inverse transform predictions
    if test_data.scale:
        try:
            preds_inverse = np.zeros_like(preds)
            trues_inverse = np.zeros_like(trues)
            for i in range(preds.shape[0]):
                for j in range(preds.shape[1]):
                    pred_slice = np.expand_dims(preds[i, j, :], axis=0)
                    true_slice = np.expand_dims(trues[i, j, :], axis=0)
                    preds_inverse[i, j, :] = test_data.inverse_transform(pred_slice)
                    trues_inverse[i, j, :] = test_data.inverse_transform(true_slice)
        except Exception as e:
            print(f"Error during inverse transform: {e}")
            preds_inverse = preds
            trues_inverse = trues
    else:
        preds_inverse = preds
        trues_inverse = trues

    # Create visualizations for each holiday
    for holiday_key, holiday_data_item in individual_holiday_windows.items():
        best_window = holiday_data_item["best_window"]
        matching_holiday = None
        for h in best_window["holidays"]:
            h_key = f"{h['holiday_name']}_{h['holiday_start'].year}"
            if h_key == holiday_key:
                matching_holiday = h
                break

        if not matching_holiday:
            continue

        window_dates = best_window["window_dates"]
        data_idx = best_window["window_idx"]

        holiday_name = matching_holiday["holiday_name"]
        holiday_start = matching_holiday["holiday_start"]
        holiday_end = matching_holiday["holiday_end"]
        holiday_year = holiday_start.year

        if data_idx >= preds_inverse.shape[0]:
            continue

        holiday_mask = (window_dates >= holiday_start) & (window_dates <= holiday_end)
        holiday_indices = np.where(holiday_mask)[0]

        if len(holiday_indices) == 0:
            continue

        for feature_idx in range(preds_inverse.shape[-1]):
            try:
                plt.figure(figsize=(18, 10))

                pred_values = preds_inverse[data_idx, :, feature_idx]
                true_values = trues_inverse[data_idx, :, feature_idx]

                pred_trend = extract_central_tendency(pd.Series(pred_values), window_hours=48)
                true_trend = extract_central_tendency(pd.Series(true_values), window_hours=48)

                dates_array = window_dates.values
                dates_matplotlib = mdates.date2num(dates_array)
                holiday_dates_matplotlib = mdates.date2num(
                    window_dates.iloc[holiday_indices].values
                )
                date_format = mdates.DateFormatter("%Y-%m-%d-%H")

                # Main plot
                ax1 = plt.subplot(2, 1, 1)
                plt.plot(dates_matplotlib, true_values, label="Ground Truth",
                         color="#4A90E2", linewidth=2, alpha=0.7)
                plt.plot(dates_matplotlib, pred_values, label="Prediction",
                         color="#E85D75", linewidth=2, linestyle="--", alpha=0.7)
                plt.plot(dates_matplotlib, true_trend, label="Ground Truth (Central Trend)",
                         color="#1F4E79", linewidth=4, linestyle="-", alpha=0.9)
                plt.plot(dates_matplotlib, pred_trend, label="Prediction (Central Trend)",
                         color="#B91C3A", linewidth=4, linestyle=":", alpha=0.9)

                plt.axvspan(holiday_dates_matplotlib[0], holiday_dates_matplotlib[-1],
                            alpha=0.2, color="yellow",
                            label=f"Holiday Period ({holiday_name} {holiday_year})")

                plt.gca().xaxis.set_major_formatter(date_format)
                tick_positions = [dates_matplotlib[0]]
                step_size = min(336, len(dates_matplotlib) // 4)
                for idx in range(step_size, len(dates_matplotlib) - step_size, step_size):
                    if idx < len(dates_matplotlib):
                        tick_positions.append(dates_matplotlib[idx])
                tick_positions.append(dates_matplotlib[-1])
                plt.xticks(tick_positions, rotation=45)

                plt.title(f"City {feature_idx+1}: Prediction during {holiday_name} {holiday_year}")
                plt.legend(fontsize=9)
                plt.grid(True, linestyle="--", alpha=0.7)

                # Zoomed-in plot
                ax2 = plt.subplot(2, 1, 2)
                plt.plot(holiday_dates_matplotlib, true_values[holiday_indices],
                         label="Ground Truth", color="blue", linewidth=3)
                plt.plot(holiday_dates_matplotlib, pred_values[holiday_indices],
                         label="Prediction", color="red", linewidth=3, linestyle="--")

                plt.gca().xaxis.set_major_formatter(date_format)
                plt.title(f"Magnified View of {holiday_name} {holiday_year} Period")
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.tight_layout()

                save_name = (
                    f"holiday_predictions/city_{feature_idx+1}_"
                    f"{holiday_name.replace(' ', '_')}_{holiday_year}_"
                    f"model_{args.model}_seq{args.seq_len}_pred{args.pred_len}.pdf"
                )
                plt.savefig(save_name)
                plt.close()

                print(f"Created visualization for City {feature_idx+1}, {holiday_name} {holiday_year}")

            except Exception as e:
                print(f"Error creating plot for feature {feature_idx+1}: {e}")
                plt.close()

    print("Holiday visualization complete.")
    return individual_holiday_windows

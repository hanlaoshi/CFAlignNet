import argparse
import os
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
import time
import random
import numpy as np

from utils.tools import (
    del_files,
    EarlyStopping,
    adjust_learning_rate,
    vali,
    load_content,
    test,
    visualize_holiday_predictions,
)
from utils.metrics import metric

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="CFAlignNet for Time Series Forecasting")

# Basic config
parser.add_argument("--task_name", type=str, required=True, default="long_term_forecast",
                    help="Task name, options: [long_term_forecast, short_term_forecast]")
parser.add_argument("--is_training", type=int, required=True, default=1, help="Training status")
parser.add_argument("--model_id", type=str, required=True, default="test", help="Model id")
parser.add_argument("--model", type=str, required=True, default="CFAlignNet",
                    help="Model name, options: [CFAlignNet]")

# Data loader
parser.add_argument("--data", type=str, required=True, default="ETTh1", help="Dataset type")
parser.add_argument("--root_path", type=str, default="./dataset/", help="Root path of the data file")
parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="Data file name")
parser.add_argument("--features", type=str, default="M",
                    help="Forecasting task, options: [M, S, MS]; "
                    "M: multivariate, S: univariate, MS: multivariate predict univariate")
parser.add_argument("--target", type=str, default="OT", help="Target feature in S or MS task")
parser.add_argument("--freq", type=str, default="h",
                    help="Frequency for time features encoding, options: [s, t, h, d, b, w, m]")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Location of model checkpoints")

# Forecasting task
parser.add_argument("--seq_len", type=int, default=192, help="Input sequence length")
parser.add_argument("--label_len", type=int, default=96, help="Start token length")
parser.add_argument("--pred_len", type=int, default=720, help="Prediction sequence length")
parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="Subset for M4")

# Model configuration
parser.add_argument("--enc_in", type=int, default=7, help="Encoder input size")
parser.add_argument("--dec_in", type=int, default=7, help="Decoder input size")
parser.add_argument("--c_out", type=int, default=7, help="Output size")
parser.add_argument("--d_model", type=int, default=16, help="Dimension of model")
parser.add_argument("--n_heads", type=int, default=8, help="Number of heads")
parser.add_argument("--e_layers", type=int, default=2, help="Number of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="Number of decoder layers")
parser.add_argument("--d_ff", type=int, default=32, help="Dimension of fcn")
parser.add_argument("--moving_avg", type=int, default=25, help="Window size of moving average")
parser.add_argument("--factor", type=int, default=1, help="Attention factor")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
parser.add_argument("--embed", type=str, default="timeF",
                    help="Time features encoding, options: [timeF, fixed, learned]")
parser.add_argument("--activation", type=str, default="gelu", help="Activation function")
parser.add_argument("--output_attention", action="store_true", help="Whether to output attention in encoder")
parser.add_argument("--patch_len", type=int, default=16, help="Patch length")
parser.add_argument("--stride", type=int, default=8, help="Stride")
parser.add_argument("--prompt_domain", type=int, default=0, help="Use domain-specific prompts")
parser.add_argument("--num_tokens", type=int, default=1000, help="Number of mapped tokens")

# LLM configuration
parser.add_argument("--llm_model", type=str, default="GPT2",
                    help="LLM model, options: [LLAMA, LLAMA3_2, GPT2]")
parser.add_argument("--llm_dim", type=int, default=768, help="LLM model dimension")
parser.add_argument("--llm_layers", type=int, default=6, help="Number of LLM layers to use")

# LoRA configuration
parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

# Holiday data
parser.add_argument("--holiday_data_path", type=str, default="",
                    help="Path to holiday data Excel file")

# Optimization
parser.add_argument("--num_workers", type=int, default=10, help="Data loader num workers")
parser.add_argument("--itr", type=int, default=1, help="Experiments times")
parser.add_argument("--train_epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size of train input data")
parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size of evaluation")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Optimizer learning rate")
parser.add_argument("--des", type=str, default="test", help="Experiment description")
parser.add_argument("--loss", type=str, default="MSE", help="Loss function")
parser.add_argument("--lradj", type=str, default="constant", help="Adjust learning rate")
parser.add_argument("--pct_start", type=float, default=0.2, help="pct_start")
parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision", default=False)
parser.add_argument("--percent", type=int, default=100, help="Percentage of training data")
parser.add_argument("--model_comment", type=str, default="none", help="Model comment/tag for logging")

args = parser.parse_args()

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

for ii in range(args.itr):
    # Setting record of experiments
    setting = (
        "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}"
    ).format(
        args.task_name,
        args.model_id,
        args.model,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des,
        ii,
    )

    # Load data
    from data_provider.data_factory import data_provider

    train_data, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")

    if args.prompt_domain:
        args.content = load_content(args)

    # Build model
    if args.model == "CFAlignNet":
        from models.CFAlignNet import Model
    else:
        raise ValueError(f"Unknown model: {args.model}. Only CFAlignNet is supported.")

    model = Model(args).float()

    path = os.path.join(args.checkpoints, setting + "-" + args.model_comment)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = torch.optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == "TST":
        train_steps = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )
    else:
        scheduler = None

    criterion = torch.nn.MSELoss()
    mae_metric = torch.nn.L1Loss()
    mape_metric = torch.nn.L1Loss()

    (
        train_loader,
        vali_loader,
        test_loader,
        model,
        model_optim,
    ) = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim
    )

    if args.is_training:
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
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
                (
                    outputs,
                    alignment_loss,
                    source_embeddings_out,
                    enc_out_patch,
                    enc_out_reprog,
                    gate_weights,
                ) = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len :, f_dim:]
                batch_y = batch_y[:, -args.pred_len :, f_dim:]

                loss = criterion(outputs, batch_y) + alignment_loss

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (args.train_epochs - epoch) * train_steps - i
                    )
                    accelerator.print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                accelerator.backward(loss)
                model_optim.step()

                if args.lradj == "TST":
                    adjust_learning_rate(
                        accelerator,
                        model_optim,
                        scheduler,
                        epoch + 1,
                        args,
                        printout=False,
                    )
                    scheduler.step()

            accelerator.print(
                "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
            )
            train_loss = np.average(train_loss)
            vali_loss, vali_mae_loss, vali_mape_loss = vali(
                args,
                accelerator,
                model,
                vali_data,
                vali_loader,
                criterion,
                mae_metric,
                mape_metric,
            )
            test_loss, test_mae_loss, test_mape_loss = vali(
                args,
                accelerator,
                model,
                test_data,
                test_loader,
                criterion,
                mae_metric,
                mape_metric,
            )

            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Vali MAE: {3:.7f} "
                "Test Loss: {4:.7f} Test MAE: {5:.7f}".format(
                    epoch + 1,
                    train_loss,
                    vali_loss,
                    vali_mae_loss,
                    test_loss,
                    test_mae_loss,
                )
            )

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != "TST":
                adjust_learning_rate(
                    accelerator, model_optim, scheduler, epoch + 1, args
                )
            else:
                accelerator.print(
                    "Updating learning rate to {}".format(
                        scheduler.get_last_lr()[0]
                    )
                )

    # Testing
    accelerator.print("Loading best model for testing...")
    model = accelerator.unwrap_model(model)
    model.load_state_dict(
        torch.load(os.path.join(path, "checkpoint.pth"), map_location="cpu"),
        strict=False,
    )

    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(accelerator.device)
            )

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
            batch_y = batch_y[:, -args.pred_len :, f_dim:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    mae, mse = metric(preds, trues)
    accelerator.print("mse:{}, mae:{}".format(mse, mae))

    # Save results
    result_path = "./results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(os.path.join(result_path, "result.txt"), "a") as f:
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n\n")

    np.save(
        os.path.join(result_path, f"{setting}_pred.npy"), preds
    )
    np.save(
        os.path.join(result_path, f"{setting}_true.npy"), trues
    )

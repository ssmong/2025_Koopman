import os
import logging
import random
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import wandb

from src.utils.plot import plot_trajectory
from src.utils.load import load_finetune_model

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("max", lambda *args: max(args))

@hydra.main(config_path="../config", config_name="learning.yaml", version_base=None)
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    #       0. WandB & Output Directory
    # ------------------------------------------------------------------
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    orig_cwd = hydra.utils.get_original_cwd()
    rel_path = os.path.relpath(output_dir, orig_cwd)

    run_name = rel_path.replace(os.path.sep, "_")
    if run_name.startswith("outputs_learning_"):
        run_name = run_name[len("outputs_learning_"):]

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=wandb_cfg,
        name=run_name,
        id=run_name
    )

    # ------------------------------------------------------------------
    #       1. Dataset & DataLoader
    # ------------------------------------------------------------------
    log.info(f"Initializing dataset...")

    train_steps = cfg.train.train_steps
    val_steps = cfg.train.val_steps
    test_steps = cfg.train.test_steps
    n_step_max = max(train_steps, val_steps, test_steps)

    if cfg.train.get("pretrained_dir"):
        pretrained_dir = cfg.train.pretrained_dir
        stats_file_path = os.path.join("outputs", "learning", pretrained_dir, "stats", "attitude_stats.json")
        stats_dir = os.path.dirname(stats_file_path) 

        if not os.path.exists(stats_file_path):
            raise FileNotFoundError(f"Pretrained stats file not found at {stats_file_path}")
        
        log.info(f"Using pretrained stats from {stats_file_path}")
    else:
        stats_dir = os.path.join(output_dir, "stats")

    train_dataset = hydra.utils.instantiate(
        cfg.data, 
        split="train", 
        train_steps=train_steps,
        stats_dir=stats_dir
    )
    val_dataset = hydra.utils.instantiate(
        cfg.data, 
        split="val", 
        train_steps=val_steps,
        stats_dir=stats_dir
    )

    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True, 
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        drop_last=False,
        pin_memory=True,
        persistent_workers=True
    )

    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    #       2. Model & Optimizer & Scheduler & Loss
    # ------------------------------------------------------------------
    log.info(f"Initializing model...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available. Please check your GPU configuration.")

    if cfg.train.get("pretrained_dir"):
        log.info(f"Fine-tuning mode: Loading from outputs/learning/{cfg.train.pretrained_dir}")
        model = load_finetune_model(
            cfg=cfg, 
            pretrained_dir=cfg.train.pretrained_dir, 
            device=device,
            strict=False 
        )
    else:
        model = hydra.utils.instantiate(cfg.model)
        model.to(device)
    
    criterion = hydra.utils.instantiate(cfg.loss, n_step_max=n_step_max)
    criterion.bind_model(model)
    criterion.to(device)
    
    combined_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = hydra.utils.instantiate(cfg.train.optimizer, params=combined_params)

    # Build scheduler
    scheduler_cfg = OmegaConf.to_container(cfg.train.scheduler, resolve=True)
    step_per = scheduler_cfg.pop('step_per', 'epoch') # Pop out step_per key

    if step_per == 'batch':
        steps_per_epoch = len(train_loader)
        if 'T_0' in scheduler_cfg:
            orig_t0 = scheduler_cfg['T_0']
            new_t0 = int(orig_t0 * steps_per_epoch)
            scheduler_cfg['T_0'] = new_t0
            log.info(f"Scaled T_0: {orig_t0} epochs -> {new_t0} steps")
        if 'T_max' in scheduler_cfg:
            scheduler_cfg['T_max'] = int(cfg.train.epochs * steps_per_epoch)

    scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)
    
    scaler = torch.amp.GradScaler()

    # ------------------------------------------------------------------
    #       3. Training Loop
    # ------------------------------------------------------------------
    log.info(f"Starting training...")

    epochs = cfg.train.epochs
    early_stopping = hydra.utils.instantiate(cfg.train.callbacks)

    best_model_path = os.path.join(output_dir, "best_model.pt")
    early_stopping.set_path(best_model_path)

    seq_strategy = cfg.train.seq_strategy

    warmup_epochs = seq_strategy.warmup_epochs
    warmup_steps = seq_strategy.warmup_steps
    min_steps = seq_strategy.min_steps
    max_steps = seq_strategy.max_steps

    if max_steps > train_steps:
        raise ValueError(f"Max steps ({max_steps}) cannot be greater than dataset prediction length ({train_steps})")
    
    global_step = 0 # For logging WandB

    for epoch in range(epochs):
        epoch_start_time = time.time()
        # ------ TRAINING ----------------------------------------------
        model.train()

        train_metrics_sum = {}

        if epoch < warmup_epochs:
            mode_desc = "Warmup"
        else:
            mode_desc = "Random"
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{mode_desc}]")
        
        for batch_idx, batch in enumerate(pbar):
            if epoch < warmup_epochs:
                curr_steps = warmup_steps
            else:
                curr_steps = random.randint(min_steps, max_steps)

            batch_gpu = {}    
            for k, v in batch.items():
                tensor_gpu = v.to(device, non_blocking=True)
                if k in ['x_future', 'u_future']:
                    batch_gpu[k] = tensor_gpu[:, :curr_steps, :]
                else:
                    batch_gpu[k] = tensor_gpu

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(enabled=True, device_type=device.type):
                results = model(n_steps=curr_steps, **batch_gpu)
                loss, metrics = criterion(results)

            loss_val = loss.item()
            
            if not torch.isfinite(loss):
                log.warning(f"Batch {batch_idx}: Loss is {loss_val} (NaN/Inf). Skipping batch.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if cfg.train.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()

            if step_per == 'batch':
                scheduler.step()

            step_log = {f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
            step_log['epoch'] = epoch
            step_log['train/grad_norm'] = total_norm
            step_log['train/lr'] = optimizer.param_groups[0]['lr']
            wandb.log(step_log, step=global_step)
            global_step += 1

            for k, v in metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                train_metrics_sum[k] = train_metrics_sum.get(k, 0.0) + val
            
            current_total_loss_sum = train_metrics_sum.get('loss/total', 0.0)
            running_avg_loss = current_total_loss_sum / (batch_idx + 1)
            pbar.set_postfix({'avg_loss': f"{running_avg_loss:.6f}"})

        avg_train_metrics = {k: v / len(train_loader) for k, v in train_metrics_sum.items()}

        # ------ VALIDATION ----------------------------------------------
        model.eval()
        val_metrics_sum = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with torch.amp.autocast(enabled=True, device_type=device.type):
                    results = model(n_steps=test_steps, **batch)
                    _, metrics = criterion(results)
                
                for k, v in metrics.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + val
        
        avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}

        # ------ LOGGING ----------------------------------------------
        epoch_log = {}

        for k, v in avg_val_metrics.items():
            epoch_log[f"val/{k}"] = v

        epoch_log['epoch'] = epoch + 1
        epoch_log['epoch_time'] = time.time() - epoch_start_time

        epoch_log.update(avg_train_metrics)
        epoch_log.update(avg_val_metrics)
        
        wandb.log(epoch_log, step=global_step)

        log_msg = [f"\nEpoch {epoch+1}/{epochs} Summary:"]
        
        # Sort keys (total to the front)
        all_keys = sorted(set(avg_train_metrics.keys()) | set(avg_val_metrics.keys()))
        if 'loss/total' in all_keys:
            all_keys.remove('loss/total')
            all_keys.insert(0, 'loss/total')

        metric_width = 15
        val_width = 12

        header = f"{'Type':<{metric_width}} | " + " | ".join([f"{k.replace('loss/', ''):<{val_width}}" for k in all_keys])
        log_msg.append(header)
        log_msg.append("-" * len(header))

        train_row = f"{'Train':<{metric_width}} | "
        train_vals = []
        for k in all_keys:
            val = avg_train_metrics.get(k, float('nan'))
            train_vals.append(f"{val:<{val_width}.6f}")
        train_row += " | ".join(train_vals)
        log_msg.append(train_row)

        val_row = f"{'Val':<{metric_width}} | "
        val_vals = []
        for k in all_keys:
            val = avg_val_metrics.get(k, float('nan'))
            val_vals.append(f"{val:<{val_width}.6f}")
        val_row += " | ".join(val_vals)
        log_msg.append(val_row)
            
        log.info("\n".join(log_msg))

        # ------ SCHEDULER & EARLY STOPPING --------------------------------
        val_total_loss = avg_val_metrics.get("loss/total", float('inf')) 
        if step_per == 'epoch':
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_metrics.get("loss/total", float('inf')))
            else:
                scheduler.step()
            
        early_stopping(val_total_loss, model)
        if early_stopping.early_stop:
            log.info("Early stopping triggered.")
            break

    # ------------------------------------------------------------------
    #       4. Test Evaluation
    # ------------------------------------------------------------------
    log.info("\n" + "="*50)
    log.info("Starting Testing with Best Model...")
    log.info("="*50)

    if os.path.exists(best_model_path):
        log.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        log.warning(f"Best model not found at {best_model_path}. Using current model weights.")

    test_dataset = hydra.utils.instantiate(cfg.data, split="test", train_steps=test_steps)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.train.batch_size, 
        num_workers=cfg.train.num_workers, 
        shuffle=False, 
        drop_last=False,
        pin_memory=True
    )

    log.info(f"Test batches: {len(test_loader)}")

    model.eval()
    test_metrics_sum = {}    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(enabled=True, device_type=device.type):
                results = model(n_steps=test_steps, **batch)
                _, metrics = criterion(results)
            
            for k, v in metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                test_metrics_sum[k] = test_metrics_sum.get(k, 0.0) + val

    avg_test_metrics = {k: v / len(test_loader) for k, v in test_metrics_sum.items()}

    wandb_test_log = {f"test/{k}": v for k, v in avg_test_metrics.items()}
    wandb.log(wandb_test_log)

    log_msg = ["\nFinal Test Results:"]
    log_msg.append(f"{'Metric':<25} | {'Test Score':<12}")
    log_msg.append("-" * 40)

    all_keys = sorted(avg_test_metrics.keys())
    if 'loss/total' in all_keys:
        all_keys.remove('loss/total')
        all_keys.insert(0, 'loss/total')

    for k in all_keys:
        val = avg_test_metrics[k]
        k_disp = k.replace("loss/", "")
        log_msg.append(f"{k_disp:<25} | {val:<12.6f}")

    log.info("\n".join(log_msg))
    log.info("Training and Testing Completed.")

    test_loader_plot = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader_plot):
            with torch.amp.autocast(enabled=True, device_type=device.type):
                batch = {k: v.to(device) for k, v in batch.items()}
                results = model(n_steps=test_steps, **batch)
            
            if i == 0:
                x_pred = results['x_traj'][0] 
                x_gt = results['x_traj_gt'][0]
                
                plot_save_path = os.path.join(output_dir, "test_plot.png")
                plot_trajectory(
                    pred_dt=cfg.data.pred_dt,
                    x_pred=x_pred,
                    x_gt=x_gt,
                    angle_indices=cfg.data.angle_indices,
                    quat_indices=cfg.data.quat_indices,
                    save_path=plot_save_path,
                    mean=train_dataset.processor.mean,
                    std=train_dataset.processor.std
                )
                
                wandb.log({"test/trajectory_plot": wandb.Image(plot_save_path)})
                log.info(f"Test plot uploaded to WandB and saved to {plot_save_path}")
                break

    wandb.finish()

if __name__ == "__main__":
    main()

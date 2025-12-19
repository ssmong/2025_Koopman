import os
import logging
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, HydraConfig

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    #       1. Dataset & DataLoader
    # ------------------------------------------------------------------
    log.info(f"Initializing dataset...")

    train_dataset = hydra.utils.instantiate(cfg.data, split="train")
    val_dataset = hydra.utils.instantiate(cfg.data, split="val")

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
    model = hydra.utils.instantiate(cfg.model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
    else:
        raise RuntimeError("CUDA is not available. Please check your GPU configuration.")
    
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    criterion = hydra.utils.instantiate(cfg.loss)
    criterion.bind_model(model)

    # ------------------------------------------------------------------
    #       3. Training Loop
    # ------------------------------------------------------------------
    log.info(f"Starting training...")

    epochs = cfg.train.epochs
    early_stopping = hydra.utils.instantiate(cfg.callbacks.early_stopping)

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    best_model_path = os.path.join(output_dir, "best_model.pt")
    early_stopping.set_path(best_model_path)

    # Random prediction length strategy
    dataset_pred_len = cfg.data.pred_len
    seq_strategy = cfg.train.seq_strategy

    warmup_epochs = seq_strategy.warmup_epochs
    warmup_steps = seq_strategy.warmup_steps
    min_steps = seq_strategy.min_steps
    max_steps = seq_strategy.max_steps

    if max_steps > dataset_pred_len:
        raise ValueError(f"Max steps ({max_steps}) cannot be greater than dataset prediction length ({dataset_pred_len})")

    for epoch in range(epochs):
        # ------ TRAINING ----------------------------------------------
        model.train()

        train_metrics_sum = {}

        if epoch < warmup_epochs:
            mode_desc = "Warmup"
        else:
            mode_desc = "Random"
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{mode_desc}]")
        
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            if epoch < warmup_epochs:
                curr_steps = warmup_steps
            else:
                curr_steps = random.randint(min_steps, max_steps)

            batch_gpu = {}    
            for k, v in batch.items():
                if k in ['x_future', 'u_future']: 
                    batch_gpu[k] = v[:, :curr_steps, :].to(device, non_blocking=True)
                else:
                    batch_gpu[k] = v.to(device, non_blocking=True)

            results = model(n_steps=curr_steps, **batch_gpu)
            loss, metrics = criterion(results)

            optimizer.zero_grad()
            loss.backward()
            if cfg.train.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()

            for k, v in metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                train_metrics_sum[k] = train_metrics_sum.get(k, 0.0) + val
            
            total_loss = metrics.get('loss/total', loss).item()
            pbar.set_postfix({'loss': f"{total_loss:.6f}"})

        avg_train_metrics = {k: v / len(train_loader) for k, v in train_metrics_sum.items()}

        # ------ VALIDATION ----------------------------------------------
        model.eval()
        val_metrics_sum = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                results = model(n_steps=dataset_pred_len, **batch)
                _, metrics = criterion(results)
                
                for k, v in metrics.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + val
        
        avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}

        # ------ LOGGING ----------------------------------------------
        log_msg = [f"\nEpoch {epoch+1}/{epochs} Summary:"]
        log_msg.append(f"{'Metric':<25} | {'Train':<12} | {'Val':<12}")
        log_msg.append("-" * 55)
        
        # 키 정렬 (total을 맨 앞으로)
        all_keys = sorted(set(avg_train_metrics.keys()) | set(avg_val_metrics.keys()))
        if 'loss/total' in all_keys:
            all_keys.remove('loss/total')
            all_keys.insert(0, 'loss/total')

        for k in all_keys:
            t_val = avg_train_metrics.get(k, float('nan'))
            v_val = avg_val_metrics.get(k, float('nan'))
            k_disp = k.replace("loss/", "")
            log_msg.append(f"{k_disp:<25} | {t_val:<12.6f} | {v_val:<12.6f}")
            
        log.info("\n".join(log_msg))

        # ------ SCHEDULER & EARLY STOPPING --------------------------------
        val_total_loss = avg_val_metrics.get("loss/total", float('inf'))
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_total_loss)
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

    test_dataset = hydra.utils.instantiate(cfg.data, split="test")
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.train.batch_size, 
        num_workers=cfg.train.num_workers, 
        shuffle=False, 
        drop_last=False,
        pin_memory=True
    )

    log.info(f"Test batches: {len(test_loader)}")

    # Test Loop
    model.eval()
    test_metrics_sum = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            results = model(n_steps=dataset_pred_len, **batch)
            _, metrics = criterion(results)
            
            for k, v in metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                test_metrics_sum[k] = test_metrics_sum.get(k, 0.0) + val

    # Logging Results
    avg_test_metrics = {k: v / len(test_loader) for k, v in test_metrics_sum.items()}

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
        

if __name__ == "__main__":
    main()
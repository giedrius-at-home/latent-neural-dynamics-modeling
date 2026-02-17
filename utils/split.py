from utils.config import Config
import polars as pl
from pathlib import Path
from utils.logger import get_logger


def create_splits(
    recordings: pl.DataFrame, split_params: Config, results_config: Config
):
    logger = get_logger()

    assert split_params.within_session_split, "Only within-session split is supported"

    recordings = recordings.with_columns(
        pl.col("n_epochs").cum_sum().alias("cum_sum_epochs")
    )

    total_epochs = int(recordings.select(pl.col("n_epochs").sum()).item())

    min_train = int(split_params.min_train_epochs)
    train_epochs_target = max(int(round(split_params.train * total_epochs)), min_train)
    train_epochs = min(train_epochs_target, total_epochs)

    leftover_epochs = max(total_epochs - train_epochs, 0)

    vt_sum = float(split_params.val) + float(split_params.test)
    val_ratio = float(split_params.val) / vt_sum
    val_epochs = int(round(leftover_epochs * val_ratio))
    test_epochs = leftover_epochs - val_epochs

    logger.info(
        f"Split plan (epochs): total={total_epochs}, train={train_epochs}, val={val_epochs}, test={test_epochs}"
    )

    train_trials = recordings.filter(pl.col("cum_sum_epochs") <= train_epochs)

    val_cutoff = train_epochs + val_epochs
    val_trials = recordings.filter(
        (pl.col("cum_sum_epochs") > train_epochs)
        & (pl.col("cum_sum_epochs") <= val_cutoff)
    )
    test_trials = recordings.filter(pl.col("cum_sum_epochs") >= val_cutoff)

    train_ep = (
        int(train_trials.select(pl.col("n_epochs").sum()).item())
        if train_trials.height > 0
        else 0
    )
    val_ep = (
        int(val_trials.select(pl.col("n_epochs").sum()).item())
        if val_trials.height > 0
        else 0
    )
    test_ep = (
        int(test_trials.select(pl.col("n_epochs").sum()).item())
        if test_trials.height > 0
        else 0
    )
    logger.info(
        f"Resulting splits: train={train_trials.height} trials ({train_ep} epochs), "
        f"val={val_trials.height} trials ({val_ep} epochs), test={test_trials.height} trials ({test_ep} epochs)"
    )

    save_dir = Path(results_config.save_dir) / "split"
    save_dir.mkdir(parents=True, exist_ok=True)
    train_trials.write_parquet(save_dir / "train.parquet")
    val_trials.write_parquet(save_dir / "val.parquet")
    test_trials.write_parquet(save_dir / "test.parquet")

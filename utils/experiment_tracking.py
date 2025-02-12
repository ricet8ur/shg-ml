import numpy as np
from aim import Run
from sklearn import metrics
from scipy import stats


def calculate_metrics(
    target: np.ndarray,
    predict: np.ndarray,
) -> dict:
    mae = metrics.mean_absolute_error(target, predict)
    rmse = metrics.root_mean_squared_error(target, predict)
    r2 = metrics.r2_score(target, predict)
    spearmanr = stats.spearmanr(target, predict)
    return dict(
        mae=mae,
        rmse=rmse,
        r2=r2,
        spearmanr_statistic=spearmanr.statistic,
        spearmanr_pvalue=spearmanr.pvalue,
    )


def track_metrics(
    run: Run,
    subset: str,
    epoch: int,
    fold: int,
    loss: float,
    keys: list[str],
    predict: np.ndarray,
    target: np.ndarray,
    to_track: bool = True,
):
    test_metrics = calculate_metrics(target, predict)
    test_metrics["loss"] = loss
    if to_track:
        run.track(
            test_metrics,
            epoch=epoch,
            context=dict(subset=subset, fold=fold),
        )
    if subset == "test":
        run["test_metrics_for_all_folds"] = run.get(
            "test_metrics_for_all_folds", {}
        ) | {fold: test_metrics}
    return test_metrics


def log_mean_std_based_on_test_metrics(run: Run):
    test_metrics_for_all_folds = list(run["test_metrics_for_all_folds"].values())
    run["test_metrics_mean"] = {
        k: float(np.mean([d[k] for d in test_metrics_for_all_folds]))
        for k in test_metrics_for_all_folds[0].keys()
    }
    run["test_metrics_std"] = {
        k: float(np.std([d[k] for d in test_metrics_for_all_folds]))
        for k in test_metrics_for_all_folds[0].keys()
    }

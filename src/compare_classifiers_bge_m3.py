"""PromptLeakGuard: controlled BGE-M3 classifier comparison.

Important: use the ORIGINAL BGE-M3 embeddings, with the original row order
and WITHOUT L2 normalization, to match the final PromptLeakGuard protocol.
Never tune hyperparameters using the held-out test set or FAH60.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

SEED = 42
SCRIPT_VERSION = "2026-10-13-NORMCHECK-3"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, help='Optional project root if auto-detection fails.')
    parser.add_argument('--train-npy', type=Path)
    parser.add_argument('--test-npy', type=Path)
    parser.add_argument('--train-parquet', type=Path)
    parser.add_argument('--test-parquet', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/classifier_comparison'))
    parser.add_argument('--skip-xgboost', action='store_true')
    parser.add_argument('--expected-train', type=int, default=546)
    parser.add_argument('--expected-test', type=int, default=116)
    return parser.parse_args()


def find_project_root(explicit_root=None):
    """Find the project folder independently of the terminal working directory.

    When this script lives under PromptLeakGuard/src/, its parent folder is
    PromptLeakGuard. The function also supports execution from the project
    root, or an explicit --project-root argument.
    """
    script_folder = Path(__file__).resolve().parent
    roots = ([explicit_root] if explicit_root is not None else []) + [
        script_folder, *script_folder.parents, Path.cwd(), *Path.cwd().parents
    ]
    seen = set()
    for root in roots:
        root = Path(root).resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / 'outputs' / 'embeddings_data').is_dir() and (
                root / 'datasets' / 'promptInjection'
        ).is_dir():
            print(f'Project root: {root}')
            return root
    folders = ', '.join(str(r) for r in seen)
    raise FileNotFoundError(
        'Cannot locate a project root containing outputs/embeddings_data '
        'and datasets/promptInjection. Use --project-root. Checked: ' + folders
    )


def locate(value, candidates, project_root):
    """Find a file, resolving user paths relative to the project root.

    The candidates are real alternative filenames, not files that all need
    to exist. The first existing file is used.
    """
    if value is not None:
        candidate = value if value.is_absolute() else project_root / value
        if not candidate.is_file():
            raise FileNotFoundError(f'Specified input file not found: {candidate}')
        return candidate
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        'File not found. Tried: ' + ', '.join(map(str, candidates))
        + '\nCheck the file location under the project root printed above.'
    )


def labels(values):
    """Map the same Safe/Attack labels used in the original LR scripts."""
    mapping = {'safe': 0, 'attack': 1, 'injection': 1, '0': 0, '1': 1}
    result = []
    for v in values:
        key = str(v).strip().lower()
        if key not in mapping:
            raise ValueError(f'Unexpected class label: {v!r}')
        result.append(mapping[key])
    return np.asarray(result, dtype=int)


def load_data(a):
    project_root = find_project_root(a.project_root)
    root = project_root / 'outputs' / 'embeddings_data'
    ds = project_root / 'datasets' / 'promptInjection'
    files = {
        'train_npy': locate(a.train_npy, [root/'train_embeddings_bgeM3.npy'], project_root),
        'test_npy': locate(a.test_npy, [root/'test_embeddings_bgeM3.npy'], project_root),
        'train_parquet': locate(a.train_parquet, [ds/'train.parquet'], project_root),
        'test_parquet': locate(a.test_parquet, [ds/'test.parquet'], project_root),
    }
    for kind, path in files.items():
        print(f'{kind}: {path}')
    # No encoder call, PCA, StandardScaler, or normalization is applied here.
    x_tr = np.load(files['train_npy'], allow_pickle=False)
    x_te = np.load(files['test_npy'], allow_pickle=False)
    df_tr = pd.read_parquet(files['train_parquet'])
    df_te = pd.read_parquet(files['test_parquet'])
    for df in (df_tr, df_te):
        if not {'label', 'text'}.issubset(df.columns):
            raise ValueError('Parquet must contain label and text columns.')
        if df['text'].isna().any():
            raise ValueError('Text contains missing values.')
    y_tr, y_te = labels(df_tr.label), labels(df_te.label)
    if x_tr.ndim != 2 or x_te.ndim != 2:
        raise ValueError('Both .npy arrays must have shape [rows, dimensions].')
    if x_tr.shape != (len(y_tr), 1024) or x_te.shape != (len(y_te), 1024):
        raise ValueError('Expected 1024-dimensional BGE-M3 embeddings aligned with parquet rows.')
    if len(y_tr) != a.expected_train or len(y_te) != a.expected_test:
        raise ValueError('Train/test size does not match the expected original benchmark.')
    if not np.isfinite(x_tr).all() or not np.isfinite(x_te).all():
        raise ValueError('Embeddings contain NaN or infinity.')
    if len(np.unique(y_tr)) != 2 or len(np.unique(y_te)) != 2:
        raise ValueError('Both Safe and Attack classes are required in both splits.')
    norms = np.linalg.norm(x_tr, axis=1)
    print('Embedding shapes:', x_tr.shape, x_te.shape)
    print('Label counts train:', np.bincount(y_tr), 'test:', np.bincount(y_te))
    print('Mean L2 norm:', float(norms.mean()))
    norm_test = np.linalg.norm(x_te, axis=1)
    train_unit = bool(np.allclose(norms, 1.0, atol=1e-3))
    test_unit = bool(np.allclose(norm_test, 1.0, atol=1e-3))
    print(f'Unit-vector check: train={train_unit}, test={test_unit}')
    print(f'Test mean L2 norm: {float(norm_test.mean()):.6f}')
    if train_unit != test_unit:
        raise ValueError('Train/test embedding normalization is inconsistent. Regenerate or use matching files.')
    if train_unit:
        print('NOTICE: Saved BGE-M3 embeddings have unit length. Using them unchanged for ALL models.')
        print('NOTICE: This does NOT establish equivalence to historical unnormalized BGE-M3 runs.')
    print('WARNING: Confirm .npy rows match the EXACT parquet row order. Shape alone cannot prove alignment.')
    return x_tr.astype('float32'), x_te.astype('float32'), y_tr, y_te, df_te, files, project_root


def build_models(skip_xgboost):
    """Fixed settings. This is a controlled classifier comparison, not a tuning search."""
    models = {
        'Logistic_Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        'SVM_RBF': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED),
        'Random_Forest': RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    }
    if not skip_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError('Run pip install xgboost or use --skip-xgboost') from exc
        models['XGBoost'] = XGBClassifier(
            objective='binary:logistic', eval_metric='logloss', n_estimators=200,
            max_depth=3, learning_rate=0.05, subsample=1.0,
            colsample_bytree=1.0, random_state=SEED, n_jobs=4, tree_method='hist')
    return models


def evaluate(name, clf, x_tr, x_te, y_tr, y_te, texts, out):
    """Report both Attack-class metrics and macro averages for transparent comparison."""
    clf.fit(x_tr, y_tr)
    # SVC probability=True uses cross-validation on TRAIN data for probability calibration.
    # We use one common probability threshold (0.50) for all four classifiers.
    attack_col = list(clf.classes_).index(1)
    p_attack = clf.predict_proba(x_te)[:, attack_col]
    prediction = (p_attack >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, prediction, labels=[0, 1]).ravel()
    row = dict(model=name, accuracy=accuracy_score(y_te, prediction),
               macro_precision=precision_score(y_te, prediction, average='macro', zero_division=0),
               macro_recall=recall_score(y_te, prediction, average='macro', zero_division=0),
               macro_f1=f1_score(y_te, prediction, average='macro', zero_division=0),
               attack_precision=precision_score(y_te, prediction, pos_label=1, zero_division=0),
               attack_recall=recall_score(y_te, prediction, pos_label=1, zero_division=0),
               attack_f1=f1_score(y_te, prediction, pos_label=1, zero_division=0),
               safe_recall=recall_score(y_te, prediction, pos_label=0, zero_division=0),
               roc_auc=roc_auc_score(y_te, p_attack),
               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    # Keep per-sample evidence for future scientific review.
    pd.DataFrame({'row': np.arange(len(y_te)), 'text': texts,
                  'true_label': y_te, 'predicted_label': prediction,
                  'attack_probability': p_attack}).to_csv(out/f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
    (out/f'{name}_report.txt').write_text(classification_report(
        y_te, prediction, labels=[0, 1], target_names=['Safe', 'Attack'], zero_division=0), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, prediction, labels=[0, 1],
                                            display_labels=['Safe', 'Attack'], ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name.replace('_', ' '))
    fig.tight_layout()
    fig.savefig(out/f'{name}_confusion.png', dpi=160)
    plt.close(fig)
    # Save classifier only; do not serialize/download BGE-M3 redundantly.
    joblib.dump({'classifier': clf, 'encoder_name': 'BAAI/bge-m3',
                 'threshold': 0.5, 'embedding_input': 'existing_npy_no_transform'}, out/f'{name}.joblib')
    print(name, ':', {k: round(v, 4) for k, v in row.items() if k in ('accuracy','attack_f1','macro_f1')})
    return row


def main():
    print(f"PromptLeakGuard classifier comparison — {SCRIPT_VERSION}")
    print(f"Script path: {Path(__file__).resolve()}")
    args = arguments()
    x_tr, x_te, y_tr, y_te, test_df, files, project_root = load_data(args)
    out = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    models = build_models(args.skip_xgboost)
    rows = [evaluate(name, clf, x_tr, x_te, y_tr, y_te, test_df.text.to_numpy(), out)
            for name, clf in models.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out/'all_classifier_comparison.csv', index=False, encoding='utf-8-sig')
    print('\nAll metrics:')
    print(df.round(4).to_string(index=False))
    metadata = {'seed': SEED, 'threshold': 0.5, 'normalization': False,
                'class_mapping': {'Safe': 0, 'Attack': 1},
                'input_files': {k: str(v.resolve()) for k, v in files.items()},
                'parameters': {n: m.get_params() for n, m in models.items()}}
    (out/'experiment_manifest.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    print('\nSaved files in:', out.resolve())


if __name__ == '__main__':
    main()

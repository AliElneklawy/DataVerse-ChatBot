import joblib
import logging
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from utils import create_folder
from paths import DATASETS_DIR, MODELS_DIR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def embed(texts: list[str]):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def split_data(embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return X_train, X_test, y_train, y_test


def train(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf: RandomForestClassifier, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)

    return {
        "Accuracy": acc,
        "precision": precision,
        "recall": recall,
        # "Classification Report": clf_report
    }


def plot_evaluation(clf, X_test, y_test, multi_class=False):
    y_prob = clf.predict_proba(X_test)[:, 1] if not multi_class else None

    if not multi_class:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_prob):.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        plt.show()
        # plt.savefig(create_folder(MODELS_DIR) / "roc_prec_recall.png")


def random_search(clf, X_train, y_train, clf_type="xgb"):
    if clf_type == "xgb":
        param_grid = {
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [100, 200, 300],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "gamma": [0, 0.1, 0.2],
        }
    else:
        param_grid = {
            "bootstrap": [True, False],
            "max_depth": [5, 10, 20, 30],
            "max_features": ["sqrt", "log2", 0.33],
            "min_samples_leaf": [2, 4, 8, 16],
            "min_samples_split": [5, 10, 20, 30],
            "n_estimators": [100, 200, 300],
        }

    scoring = {"accuracy": "accuracy", "Precision": "precision", "Recall": "recall"}

    rnd_search = RandomizedSearchCV(
        clf,
        param_distributions=param_grid,
        n_iter=10,
        cv=15,
        verbose=2,
        n_jobs=-1,
        scoring=scoring,
        refit="accuracy",
    )

    rnd_search.fit(X_train, y_train)

    return (
        rnd_search.best_params_,
        rnd_search.best_score_,
        rnd_search.cv_results_,
        rnd_search.best_estimator_,
    )


def save_model(model, file_path):
    joblib.dump(model, file_path)
    logger.info(f"Model saved to {file_path}")


def save_metadata(
    model, df, metadata_path, params, train_score=None, cv_score=None, test_score=None
):
    """Save metadata about libraries and training."""
    import sys
    import json
    import sklearn
    import sentence_transformers
    from datetime import datetime

    metadata = {
        "save_date": datetime.now().isoformat(),
        "library_versions": {
            "python": sys.version,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        "model_details": {
            "class": str(model.__class__.__name__),
            "hyperparameters": params,
            "train_score": train_score,
            "cv_score": cv_score,
            "test_score": test_score,
        },
        "dataset": {
            "dataset_name": "responses.csv",
            "dataset_info": {
                "source": "manually created from our own RAG system responses.",
                "shape": df.shape,
                "column_names": df.columns.tolist(),
                "label_dist": df["label"].value_counts().to_dict(),
            },
        },
        "training_info": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "test_ratio": "0.2",
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    dataset_path = create_folder(DATASETS_DIR) / "responses.csv"
    metadata_path = create_folder(MODELS_DIR) / "metadata.json"
    model_path = MODELS_DIR / "clf.pkl"
    clf = RandomForestClassifier(random_state=42)
    # clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    df = load_dataset(dataset_path)
    embeddings = embed(df["response"].to_list())
    X_train, X_test, y_train, y_test = split_data(embeddings, df["label"])

    clf = train(clf, X_train, y_train)
    train_evaluation = evaluate(clf, X_train, y_train)
    test_evaluation = evaluate(clf, X_test, y_test)

    print(pd.DataFrame([train_evaluation]))
    print(pd.DataFrame([test_evaluation]))

    plot_evaluation(clf, X_test, y_test)

    if test_evaluation["Accuracy"] < 0.95:
        best_params, best_score, cv_results, clf = random_search(
            clf, X_train, y_train, clf_type="rf"
        )
        cv_results_df = pd.DataFrame(cv_results)

        final_evaluation = evaluate(clf, X_test, y_test)
        save_metadata(
            clf,
            df,
            metadata_path,
            best_params,
            train_score=train_evaluation["Accuracy"],
            cv_score=best_score,
            test_score=final_evaluation["Accuracy"],
        )

        print("Best parameters:", best_params)
        print("Best accuracy score:", best_score)
        # print(cv_results_df)
    else:
        final_evaluation = evaluate(clf, X_test, y_test)
        save_metadata(
            clf,
            df,
            metadata_path,
            clf.get_params(),
            train_score=train_evaluation["Accuracy"],
            test_score=final_evaluation["Accuracy"],
        )

    print("\nFinal Model Evaluation on Test Set:")
    pprint(final_evaluation)

    save_model(clf, model_path)

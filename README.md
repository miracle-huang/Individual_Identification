# Individual Identification using ECG Signals

> **Project Tagline**: 異なる感情状態における心電図（ECG）信号を用いた個人識別システム。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-Pending-lightgrey)

## 📖 プロジェクト概要 (Introduction)

本プロジェクトは、**心電図（ECG）信号**を利用して、異なる感情状態にある個人の識別を行うことを目的としています。深層学習手法に基づき、被験者が異なる種類のビデオ（面白い、退屈、リラックス、怖いなどの感情を誘発するもの）を視聴している際のECGデータを分析し、高精度の個人認証を実現します。

主な機能 (Features):
*   **多感情状態への対応**: 異なる感情を誘発するビデオ（amusing, boring, relaxed, scary）視聴時のデータを用いて訓練およびテストを行います。
*   **深層学習モデル**: **CNN (畳み込みニューラルネットワーク)** と **LSTM (長短期記憶ネットワーク)** モデルを統合し、特徴抽出と分類を行います。
*   **交差検証**: 5分割交差検証（5-fold cross-validation）メカニズムを組み込み、実験結果の信頼性を確保しています。
*   **柔軟な設定**: 設定ファイルを通じて、被験者数、タイムウィンドウサイズ、トレーニングバッチサイズなどのパラメータを簡単に調整可能です。

## 🛠 技術スタック (Tech Stack)

本プロジェクトでは主に以下のフレームワークとツールを使用しています：

*   **プログラミング言語**: Python 3
*   **深層学習フレームワーク**: TensorFlow (Keras API)
*   **データ処理**: NumPy, Pandas
*   **科学計算 & 評価**: Scikit-learn
*   **データ可視化**: Matplotlib, Seaborn
*   **ファイル操作**: OpenPyXL (Excel 操作)

## 📂 ディレクトリ構成 (Directory Structure)

```text
Individual_Identification/
├── config.py           # プロジェクト設定ファイル（被験者リスト、学習パラメータ、ビデオリスト等）
├── main.py             # メインプログラムのエントリポイント、実験の実行を担当
├── load_data/          # データ読み込みモジュール
│   ├── load.py         # データ読み込みのコアロジック
│   └── load_utils.py   # データ処理ユーティリティ関数
├── model/              # モデル定義と学習モジュール
│   ├── cnn_model.py    # CNNモデル構造定義
│   ├── lstm_model.py   # LSTMモデル構造定義
│   └── train.py        # モデル学習ロジック
├── experiment/         # 実験スクリプト（論文内の異なる実験テーブルに対応）
│   ├── run_table1.py
│   ├── ...
│   └── run_table5.py
├── data/               # (各自準備が必要) 元データディレクトリ
└── result/             # 実験結果出力ディレクトリ
```

## 🚀 クイックスタート (Quick Start)

### 前提条件 (Prerequisites)

システムに Python 3.8 以上がインストールされていることを確認してください。Anaconda や仮想環境を使用して依存関係を管理することを推奨します。

### インストール (Installation)

1.  コードベースをクローンします：
    ```bash
    git clone <repository_url>
    cd Individual_Identification
    ```

2.  依存ライブラリをインストールします：
    `requirements.txt` が提供されていないため、以下のコマンドを実行して必要なライブラリをインストールしてください：
    ```bash
    pip install tensorflow numpy pandas scikit-learn openpyxl matplotlib seaborn
    ```

### 使用方法 (Usage)

1.  **データ準備**: 指定された形式の ECG データセットを `data/` ディレクトリに配置してください。
2.  **パラメータ設定**: `config.py` 内の設定（被験者番号の範囲など）を確認し、必要に応じて修正してください。
3.  **プログラム起動**:
    メインプログラムを直接実行すると、デフォルトの実験（`run_table5` のロジック）が実行されます：
    ```bash
    python main.py
    ```
    *注意：他の実験（table1 - table4 など）を実行したい場合は、`main.py` 内のインポートと呼び出しコードを変更してください。*

## ⚙️ 設定 (Configuration)

重要な設定項目は `config.py` ファイルにあります：

*   **Subject Lists** (`subject_amount_N`): 異なる規模の被験者番号リストを定義します（例：5人、10人、30人グループ）。
*   **Training Params**:
    *   `batch_size`: バッチサイズ (デフォルト 64)。
    *   `epochs`: 学習エポック数 (デフォルト 100)。
*   **Video Lists** (`video_list_N`): 感情誘発に使用されるビデオのラベルリストを定義します。
*   **Test Params** (`test_time_list`): テストに使用するタイムセグメントの長さリストを定義します。

## 🤝 コントリビューション (Contributing)

本プロジェクトへの改善提案（Pull Request や Issue）を歓迎します！
1. 本リポジトリをフォークします。
2. 機能ブランチを作成します (`git checkout -b feature/AmazingFeature`)。
3. 変更をコミットします (`git commit -m 'Add some AmazingFeature'`)。
4. ブランチにプッシュします (`git push origin feature/AmazingFeature`)。
5. Pull Request を作成します。

## 📄 ライセンス (License)

License information pending. Please add a LICENSE file to the repository.

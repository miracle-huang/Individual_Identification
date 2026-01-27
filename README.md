# Individual Identification using ECG Signals

> **Project Tagline**: 異なる感情状態における心電図（ECG）信号を用いた個人識別システム。

[![IEEE CyberSciTech 2024](https://img.shields.io/badge/IEEE-CyberSciTech%202024-00629B)](https://cyber-science.org/2024/cyberscitech/)

## � 研究背景 (Research Background)

生体信号（ECGなど）を用いた個人識別において、被験者の**感情状態（Emotion States）** が識別精度に影響を与えることが知られています。しかし、多様な感情状態が具体的にどの程度影響するのか、また、それらの影響下で安定した識別が可能であるかについては、既存の研究では十分に明らかにされていません。本プロジェクトは、この課題に取り組むための実験的検証とモデル構築を目的としています。

## 🎯 研究目的 (Research Purpose)

本研究の主な目的は、**異なる感情状態（Amusing, Boring, Relaxed, Scaryなど）においても堅牢（Robust）な個人識別手法を確立すること**です。
具体的には、CASEデータセットに含まれる動画視聴時のECG信号を用いて、感情の変化に左右されにくい特徴抽出と識別モデル（1D-CNN等）の性能を検証します。

## 🧪 実験設定 (Experimental Setup)

### データセット
[CASE データセット](https://gitlab.com/karan-shr/case_dataset) (Continuously Annotated Signals of Emotion) を使用。
<div align="center">
  <img src="https://github.com/user-attachments/assets/dc2cc379-95ac-4a22-b136-a1770bd7c4ac" width="70%">
</div>

### 感情誘発刺激
以下の4種類の感情を誘発するビデオ視聴時のデータを対象とします。
- Amusing (面白い)
- Boring (退屈)
- Relaxed (リラックス)
- Scary (怖い)

### 交差検証 (Cross-Validation)
実験の信頼性を高めるため、**5-fold Cross-Validation** を採用しています。データセットを5分割し、学習データとテストデータをローテーションさせてモデルを評価します。
<div align="center">
  <img src="https://github.com/user-attachments/assets/0b857ff7-6538-4b1b-8a00-1cd76114c3e7" width="50%">
</div>

### 複数感情状態に基づく個体識別モデル
異なる感情タイプが個体識別の性能に与える影響を調査するため、複合的な感情状態における個体識別モデルを構築しました。本モデルでは、単一または複数の感情刺激から得られたデータを任意に選択し、個体識別モデルの学習に用いることが可能です。
<div align="center">
  <img src="https://github.com/user-attachments/assets/00c18ef8-6d36-4be9-a506-554ca1d22a99" width="50%">
</div>

## 📊 データ準備 (Data Preparation)

*   **被験者数**: 設定により最大30名の被験者データを使用 (`config.py` で調整可能)。
*   **データ前処理**:
    *   **Windowing**: ECG信号を指定された時間窓（Window Size、例: 2秒）で切り出します。
    *   **Sliding**: スライディングウィンドウ（例: 10ポイント）を用いてデータを拡張します。
    *   データ読み込みのロジックは `load_data/load.py` に実装されており、各感情ラベルに対応するビデオデータを自動的に処理します。

## 🧠 深層学習モデル (Deep Learning Model)

本プロジェクトでは、時系列データであるECG信号から有効な特徴を抽出するために **1D-CNN (One-Dimensional Convolutional Neural Network)** を採用しています。
(`model/cnn_model.py` 参照)

*   **アーキテクチャ概要**:
    1.  **Conv1D Layer 1**: 32フィルター, カーネルサイズ6, ReLU活性化関数
    2.  **MaxPooling1D**: プーリングサイズ2, ストライド2
    3.  **Conv1D Layer 2**: 64フィルター, カーネルサイズ6, ReLU活性化関数
    4.  **MaxPooling1D**: プーリングサイズ2, ストライド2
    5.  **Flatten**: 全結合層への変換
    6.  **Dense Layers**: 64ユニット -> Dropout (0.5) -> 32ユニット -> Dropout (0.5)
    7.  **Output Layer**: Softmax関数によるクラス分類（被験者数に対応）

<div align="center">
  <img src="https://github.com/user-attachments/assets/d87ee8d6-b6e8-486f-825d-7db3dc1f4d1b" width="40%">
</div>

また、比較・拡張用として **LSTM (Long Short-Term Memory)** モデルも含まれています (`model/lstm_model.py`)。

## 📈 研究結果 (Research Results)

本コードベースを実行することで、論文 "Leveraging ECG Signal for People Identification under Different Emotion States" で報告されている実験結果（Table 4〜10）を再現可能です。
主な検証結果として、提案モデルは異なる感情状態が混在するデータセットにおいても、有望な識別精度（Promising Accuracy）を達成しており、感情変動に対する堅牢性が示されています。

さらに、感情刺激が「Scary（怖い）」の場合、生理信号に基づく個体識別の精度が大幅に低下することが判明しました。これは、恐怖感情が他の感情よりも生理信号に大きな変動をもたらすことに起因すると考えられます。
<div align="center">
  <img src="https://github.com/user-attachments/assets/d32f084a-2598-4f78-82a7-aa42dc29336b" width="40%">
</div>

## 🛠 技術スタック (Tech Stack)

*   **Programming**: Python 3
*   **DL Framework**: TensorFlow (Keras)
*   **Libraries**: NumPy, Pandas, Scikit-learn, OpenPyXL, Seaborn, Matplotlib

## 📂 ディレクトリ構成 (Directory Structure)

```text
Individual_Identification/
├── config.py           # 実験設定（被験者数、エポック数、バッチサイズ等）
├── main.py             # 実験実行のメインスクリプト
├── load_data/          # データロード処理
│   └── load.py         # ECGデータの読み込みと前処理
├── model/              # モデル定義
│   ├── cnn_model.py    # 1D-CNNモデル
│   └── lstm_model.py   # LSTMモデル
├── experiment/         # 各実験シナリオの実行コード
│   └── run_table5.py   # 例: Table 5の実験スクリプト
├── data/               # データセット格納場所 (ユーザーが配置)
└── result/             # 実験結果の出力先
```

## 🚀 クイックスタート (Quick Start)

### 前提条件 (Prerequisites)
*   Python 3.8+
*   推奨: Anaconda または venv 環境

### インストール (Installation)
```bash
git clone <repository_url>
cd Individual_Identification
pip install tensorflow numpy pandas scikit-learn openpyxl matplotlib seaborn
```

### 使用方法 (Usage)
1.  `data/` ディレクトリにデータセットを配置します。
2.  `main.py` を実行して実験を開始します。
    ```bash
    python main.py
    ```
    *デフォルトでは `experiment/run_table5.py` が実行されます。他の実験を行う場合は `main.py` を編集してください。*

## ⚙️ 設定 (Configuration)
`config.py` で主要なパラメータを変更できます：
*   `subject_amount_30`: 実験対象の被験者リスト
*   `batch_size`: 64 (Default)
*   `epochs`: 100 (Default)
*   `video_list`: 各感情に対応する動画ID

## 📖 文献引用 (Citation)

本プロジェクトの成果を利用する場合は、以下の論文を引用してください：

### 📗 IEEE PICom 2024
**Leveraging ECG Signal for People Identification under Different Emotion States**

Zhiying Huang, Yuang Meng, Ao Guo, Walid Brahim, Jianhua Ma

*Proceedings of the 2024 IEEE Cyber Science and Technology Congress (CyberSciTech)*.

🔗 [https://ieeexplore.ieee.org/abstract/document/10795696](https://ieeexplore.ieee.org/abstract/document/10795696)
```
@inproceedings{huang2024leveraging,
  title={Leveraging ECG Signal for People Identification under Different Emotion States},
  author={Huang, Zhiying and Meng, Yuang and Guo, Ao and Brahim, Walid and Ma, Jianhua},
  booktitle={2024 IEEE Cyber Science and Technology Congress (CyberSciTech)},
  pages={491--495},
  year={2024},
  organization={IEEE}
}
```

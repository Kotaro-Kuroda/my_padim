# PaDiM (PyTorch)

PyTorchで実装したシンプルな非公式の[PaDiM](https://arxiv.org/abs/2011.08785)学習コードです。
公式のリポジトリは[こちら](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)
MVTec ADの学習画像から特徴分布（平均・共分散逆行列）を推定し、重みとして保存します。

## 特徴

- `PaDiM` 本体実装（`models/padim.py`）
- バックボーン切り替え対応
	- `resnet*`（`torchvision.models`）
	- `dinov2*`（`torch.hub`経由）
- MVTec AD用データローダー（`dataset/mvtec_dataset.py`）
- 学習結果の保存（`weights/<backbone>_<category>.pt`）

## 環境

- Python `3.11.6`（`.python-version`）
- 依存関係（`pyproject.toml`）
	- `torch`
	- `torchvision`
	- `tqdm`
	- `matplotlib`

## セットアップ

### uvを使う場合

```bash
uv sync
```

### pipを使う場合

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision tqdm matplotlib
```

## データセット配置

`train.py` は以下の構造を想定しています（MVTec AD）。

```text
<root_dir>/
	bottle/
		train/
			good/
				xxx.png
```

デフォルトでは `train.py` 内で次を使用します。

- `root_dir=/home/localuser/data/mvtec/`
- `category=bottle`

必要に応じて `train.py` の `root_dir` / `category` / `backbone_name` を変更してください。

## 学習実行

```bash
python train.py
```

実行後、`weights/` 配下に以下が保存されます。

- `mean`: 各空間位置の特徴平均
- `cov_inv`: 各空間位置の共分散逆行列
- `idx`: ランダムサンプリングした特徴次元インデックス

## バックボーンについて

- `ResNetBackbone` は `layer1, layer2, layer3` の特徴を結合して使用します。
- 出力次元は各層の出力チャンネル数を合計して計算します。
	- `resnet18/34`: `64 + 128 + 256 = 448`
	- `resnet50/101/...`: `256 + 512 + 1024 = 1792`
- `DINOv2Backbone` は最終トークン特徴を2次元マップに整形して使用します。


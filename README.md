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


## 学習の効率化
PaDiMでは、正常画像の特徴から平均と分散共分散行列を計算する必要があります。
しかし、すべての画像の特徴量を一度リストに保持してから統計量を計算すると、データ量が大きい場合にメモリ不足に陥りやすく、学習に用いるデータ数に限りが生じます。

この実装では、特徴量をバッチごとに処理しながら、平均と分散共分散行列を逐次更新する方式を採用しています。
これにより、全特徴量を保持せずに統計量を求められるため、メモリ使用量を抑えて、より多くのデータで学習できます。


平均の更新式には次の式を用いています。

```math
\begin{aligned}
\mu_n &= \frac{1}{n}\sum_{i=1}^{n} x_i \\
&= \frac{1}{n}\left(\sum_{i=1}^{n_1} x_i + \sum_{i=1}^{n_2} x_i\right), \quad n = n_1 + n_2 \\
&= \frac{1}{n}\left(n_1\mu_{n_1} + n_2\mu_{n_2}\right)
\end{aligned}
```

こうすることで、$n$個のデータの平均を求める際に、$n$個すべてのデータを保持しておく必要がなくなります。
$n_{1}$個のデータの平均とデータ数（$n_{1}$）を保存しておけば、次の$n_{2}$個のデータの和を計算することで、$n$個のデータの平均を求められます。

同様に、分散共分散行列の更新には次の式を用いています。

ここで、$`\mathbf{d}_1 = \boldsymbol{\mu}_{n_1} - \boldsymbol{\mu}`$、$`\mathbf{d}_2 = \boldsymbol{\mu}_{n_2} - \boldsymbol{\mu}`$ と置いています。

```math
\begin{aligned}
\mathrm{Cov}_n &= \frac{1}{n - 1} \sum_{i=1}^{n} (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^{T} \\
&=\frac{1}{n - 1} \Bigg[\sum_{i=1}^{n_{1}}(\boldsymbol{x}_{i} - \boldsymbol{\mu}_{n_{1}} +\boldsymbol{\mu}_{n_{1}} - \boldsymbol{\mu})(\boldsymbol{x}_{i} - \boldsymbol{\mu}_{n_{1}} +\boldsymbol{\mu}_{n_{1}} - \boldsymbol{\mu}) ^{T} \\
&\qquad\qquad+ \sum_{i=1}^{n_{2}}(\boldsymbol{x}_{i} - \boldsymbol{\mu}_{n_{2}} +\boldsymbol{\mu}_{n_{2}} - \boldsymbol{\mu})(\boldsymbol{x}_{i} - \boldsymbol{\mu}_{n_{2}} +\boldsymbol{\mu}_{n_{2}} - \boldsymbol{\mu}) ^{T}\Bigg]\\
&= \frac{1}{n - 1}\left((n_1 - 1)\mathrm{Cov}_{n_1} + n_1\mathbf{d}_1\mathbf{d}_1^{T} + (n_2 - 1)\mathrm{Cov}_{n_2} + n_2\mathbf{d}_2\mathbf{d}_2^{T}\right)
\end{aligned}
```
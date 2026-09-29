# 強化学習入門（社内勉強会 2026-09-28）

ML 経験があり、強化学習は初めての人向けの30分の勉強会資料です。
「強化学習は、報酬で重み付けした教師あり学習として読める」という1本の筋で、Q学習と方策勾配法をつなぎます。

| ファイル | 中身 |
| --- | --- |
| `rl_study/gridworld/` | デモ1: 迷路の Q学習。価値がゴールから広がる様子を見せる |
| `rl_study/cartpole/` | デモ2: CartPole の方策勾配法（REINFORCE）。学習前後の GIF と学習曲線を出す |
| `rl_study/style.py` | 2つのデモが共通で使う、図の色と日本語フォントの選び方 |
| `slides/build_deck.py` | スライドを生成するスクリプト |
| `slides/assets/` | デモが書き出した図と GIF。スライドはここを読む |

2つのデモは、役割ごとにファイルを分けています。

```text
rl_study/gridworld/
├── __main__.py   入口（python -m rl_study.gridworld で動く）
├── env.py        迷路そのもの。1手進めて、次のマスと報酬を返す
├── agent.py      ε-greedy と Q学習
└── draw.py       価値と経路の描画、4コマの PNG と GIF の書き出し

rl_study/cartpole/
├── __main__.py   入口（python -m rl_study.cartpole で動く）
├── policy.py     方策の MLP（4→64→2）
├── train.py      環境の作成、1エピソードの実行、REINFORCE の更新と学習ループ
├── record.py     GIF の録画、学習曲線、ウィンドウでの再生
└── play.py       ハンズオンの入口。設定を変えて学習させ、学習曲線を重ねて比べる
```

## 準備

[uv](https://docs.astral.sh/uv/) を使います。入っていなければ、先に入れてください。

```bash
# macOS と Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows（PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

リポジトリの中で次を実行すると、Python 3.12 と CPU 版の PyTorch が入ります。

```bash
uv sync
```

## デモ1: 迷路の Q学習

```bash
uv run python -m rl_study.gridworld
```

ウィンドウが開き、最初の40エピソードを1コマずつ再生します。
最初の7エピソードはゴールに着かず、価値は8エピソード目からゴールの隣のマスに付き始めます。
16エピソード目には、Q が一番大きい手を選ぶだけで最短の8手で着くようになります。
上の道は12手の遠回りなので、学習後は下の8手の道を選びます。
実際には10回に1回（ε＝0.1）無作為に動くので、50エピソード以降も大半は8〜12手でゴールに着き、約5%は X に落ちて終わります。

| オプション | 既定 | 意味 |
| --- | --- | --- |
| `--episodes` | 300 | 回すエピソード数 |
| `--seed` | 0 | 乱数の種 |
| `--save DIR` | なし | ウィンドウを出さず、`gridworld_snapshots.png` と `gridworld_q.gif` を DIR に書き出す |

核になるのは `agent.py` の `q_learning` の中の2行です。

```python
target = step.reward + (0.0 if step.done else gamma * q[step.next_cell].max())
q[cell][action] += alpha * (target - q[cell][action])
```

## デモ2: CartPole の方策勾配法

```bash
uv run python -u -m rl_study.cartpole --watch
```

学習前の方策で1回動かして GIF にし、直近20エピソードの平均が475に届くまで学習して、もう1回動かします。
`--watch` を付けると、最後に学習後の方策をウィンドウで動かします。
seed 0 では、学習前は14ステップで倒れ、328エピソードの学習で500ステップ（満点）立ちます。
Mac の CPU で10秒ほどかかります。

| オプション | 既定 | 意味 |
| --- | --- | --- |
| `--updates` | 200 | 更新回数の上限（1回の更新に8エピソード使う） |
| `--seed` | 0 | 乱数の種 |
| `--out DIR` | `slides/assets` | GIF、学習曲線、`cartpole_summary.json` の書き出し先 |
| `--watch` | なし | 学習後にウィンドウで動かす |

核になるのは `train.py` の `update` の中の3行です。自分が選んだ行動を正解ラベルにした cross-entropy を、A で重み付けしています。

```python
advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
ce = F.cross_entropy(logits, actions, reduction="none")
loss = (ce * advantages).mean()
```

## ハンズオン: 設定を変えて CartPole を学習させる

```bash
uv run python -m rl_study.cartpole.play
uv run python -m rl_study.cartpole.play --lr 0.1
```

学習中は学習曲線がウィンドウで伸びていき、終わると学習後の方策で棒が動きます。
前に回した設定の曲線も重ねて描くので、値を1つずつ変えて比べてください。
記録は `runs/` に残ります。1回20秒ほどです。

| オプション | 既定 | 意味 |
| --- | --- | --- |
| `--lr` | 0.01 | 学習率 |
| `--gamma` | 0.99 | 割引率 γ |
| `--batch` | 8 | 1回の更新に使うエピソード数 |
| `--no-baseline` | なし | 平均を引かず、G をそのまま重みにする |
| `--episodes` | 600 | 学習に使うエピソード数。`--batch` で割り切れないときは切り下げる |
| `--seed` | 0 | 乱数の種 |
| `--clear` | なし | 重ねて描く過去の実行を消してから始める |

seed 0〜2 で600エピソードずつ回すと、次のようになりました。
「解けた」は、更新の区切りで見た直近20回の平均が475ステップに届いたことを指します。

| 設定 | 解けたエピソード | 最後の20回の平均ステップ数 |
| --- | --- | --- |
| 既定 | 328、336、416 | 477、421、461 |
| `--lr 0.1` | 解けない | 143、212、261 |
| `--lr 0.001` | 解けない | 51、56、57 |
| `--gamma 0.9` | 552、344、解けない | 465、138、314 |
| `--batch 32` | 解けない | 129、182、201 |
| `--no-baseline` | 352、456、536 | 460、209、489 |

## スライドを作り直す

図と GIF を作り直してから、スライドを生成します。

```bash
uv run python -m rl_study.gridworld --save slides/assets
uv run python -u -m rl_study.cartpole --out slides/assets
uv run --group slides python slides/build_deck.py
```

`build_deck.py` は `強化学習入門.pptx` を毎回丸ごと作り直します。
既存のファイルは `slides/_backup/` に日時付きで退避しますが、PowerPoint で手直ししたノートや文言は新しいファイルには残りません。
手で直した後に作り直すときは、先に `build_deck.py` 側へ同じ修正を入れてください。

14枚目のタイトルの数字（学習前のステップ数とエピソード数）は `slides/assets/cartpole_summary.json` から入ります。

### LibreOffice で PDF にして確かめる

Homebrew の LibreOffice では、fontconfig の設定を渡さないと日本語が空白になります。

```bash
FONTCONFIG_FILE=/opt/homebrew/etc/fonts/fonts.conf \
  soffice --headless --convert-to pdf --outdir slides/_render 強化学習入門.pptx
```

## 開発

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run ty check
```

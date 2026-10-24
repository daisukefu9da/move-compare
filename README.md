# ダンス比較ツール (move-compare)

お手本とレビュー対象者のダンス動画を骨格推定で比較し、「どこを・どう直せばよいか」を
**「0:32.4 付近：右肘が手本より約25°深く曲がっている」** のような形で、ズレの大きい瞬間をスロー動画で見せるツールです。

- 2動画モード: お手本の動画とレビュー対象の動画を別々に読み込む
- 1動画モード: 1 本の動画に映る複数人から「お手本」と「レビュー対象」を選ぶ
- 出力: スコア、ズレの大きい箇所の一覧、骨格を重ねた比較動画、指摘文（ルールベース / 任意で Gemini）

現状は自分用のローカルツールです。将来の Web / スマホ展開に備えて、解析ロジックは UI・推論エンジンから切り離しています（後述）。

---

## セットアップ

Python 3.11 以上、git、ffmpeg（音声の抽出と比較動画の書き出しに使用）が必要です。

```bash
# 最初の 1 回だけ: 好きな場所に取ってくる
git clone -b claude/beautiful-knuth-hfbfqy https://github.com/daisukefu9da/move-compare.git
# macOS: brew install ffmpeg / Windows: winget install ffmpeg
```

あとは **`start.command`（Mac）または `start.bat`（Windows）をダブルクリック** するだけです。起動のたびに次を自動で行います。

1. `git pull` で GitHub の最新版を取り込む（オフラインなどで失敗しても、今の版で起動）
2. 初回は仮想環境 `.venv` を作り、必要なパッケージをインストール（数分〜十数分）
3. `requirements.txt` が更新されていれば、パッケージを入れ直す
4. 画面（Streamlit）を起動

> Mac で初回に「開発元を確認できない」と出たら、`start.command` を右クリック →「開く」で起動してください。
> テスト動画はフォルダ内の `videos/` に置けば GitHub には送られません（取り込みの邪魔にもなりません）。

手動で準備する場合:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

インストールの確認だけなら、合成した棒人間の動画で一連の処理を試せます（骨格推定は使いません）:

```bash
python -m dance_compare demo --out demo
# demo/result/ に report.md, report.json, comparison.mp4 ができる
```

## 使い方

### 画面で使う（Streamlit）

```bash
streamlit run app.py
```

左のサイドバーでモード・拍の取り方・しきい値などを選び、動画をアップロード（またはパスを入力）して「解析する」。
1動画モードでは「人物を検出」→ 画像を見て（`streamlit-image-coordinates` を入れていればクリックで）お手本とレビュー対象を選びます。

**1動画モードの比較する区間**: 踊り始める前・踊り終えた後を比較に入れないよう、開始・終了を指定できます（指定した時刻のコマが表示されます）。
空欄の側は、**二人の動きがそろっている区間**から自動で推定します。「自動で推定して欄に入れる」を押すと、同期度のグラフと推定結果が出るので、確認して必要なら書き換えてください。
- 踊る前に各自でリズムを取っていても、動き方がバラバラなら振付とは区別できます
- 二人がまったく同じやり方でリズムを取っている（同じように首と肩でノる等）と、振付と区別できず開始が早めに出ます
- 「1エイト目の1」は、比較を始める時刻の直後の拍になります。ずれる場合はサイドバーで指定してください

### コマンドラインで使う

```bash
# 2動画モード
python -m dance_compare compare ref.mp4 me.mp4 --out out/
python -m dance_compare compare ref.mp4 me.mp4 --mirror          # 手本が向かい合わせで撮影されている
python -m dance_compare compare ref.mp4 me.mp4 --bpm 98 --first-one 2.35   # カウント表示を使う（BPM と「1」の位置を手で指定）
python -m dance_compare compare ref.mp4 me.mp4 --narrator gemini # Gemini で指摘を整理（要 GEMINI_API_KEY）

# 1動画モード: まず人物 ID を確認して、ID かクリック位置で指定
python -m dance_compare people group.mp4 --out people.jpg
python -m dance_compare single group.mp4 --ref-id 1 --user-id 3
python -m dance_compare single group.mp4 --ref-point 420,300 --user-point 900,320 --frame 0
python -m dance_compare single group.mp4 --ref-id 1 --user-id 3 --start 12.5          # 比較を始める時刻を指定

# 推定済みの骨格 JSON どうしを比較（動画なし）
python -m dance_compare poses out/ref_pose.json out/user_pose.json
```

主なオプション: `--ref-start/--ref-end/--user-start/--user-end`（区間指定, 秒）、`--no-audio-sync`、`--open-ends`
（レビュー対象の前後に余分な部分がある）、`--backend mediapipe`、`--config tuning.toml`、`--no-video`。

### 出力

| ファイル | 内容 |
|---|---|
| `report.md` | 指摘文（スコア・全体の傾向・優先度順の指摘・リズム）と、末尾に検出一覧の表 |
| `findings.csv` | **検出一覧**: いつ・どの部位・どの観点で・どれくらいズレたか（Excel で開ける） |
| `report.json` | すべての数値（区間ごとの値、時系列、拍、時間の対応表）。精度調整の分析用 |
| `moment_1.mp4` / `moment_1.jpg` など | **直すところ**: ズレの大きい瞬間の前後の短い動画（スロー）と静止画。お手本と自分を並べ、ズレている部位に印 |
| `comparison.mp4` | [手本][レビュー対象（ズレの大きい関節を赤丸）][正規化した骨格の重ね合わせ] + 時刻と指摘。手本の音声付き |
| `ref_pose.json` / `user_pose.json` | 選んだ人物の骨格（COCO 17 点）。`poses` コマンドで再利用可 |

骨格推定の結果は `.cache/poses/` にキャッシュされるので、しきい値を変えての再解析はすぐ終わります。

---

## 精度検証（答えがわかっているズレとの照合）

テスト動画と「期待される検出結果」を TOML のケースファイルでセットにして管理し、一括で照合できます。

```bash
cp validation/cases.example.toml validation/cases.toml    # 動画パスと期待値を書く
python -m dance_compare validate validation/cases.toml --out validation_out
# → validation_out/validation.md に、ケースごとの合否・期待ごとの結果（OK / 場所違い / 値が範囲外 / 未検出）・
#    想定外の検出（誤検出の候補）。各ケースの findings.csv も出力

python -m dance_compare make-validation-set --out validation/synth   # 合成データの見本（4 ケース）
```

ケースファイルの書き方は [validation/README.md](validation/README.md) を参照してください。

---

## 処理の流れ

1. **骨格抽出**: YOLO-pose（複数人 + ByteTrack 追跡）または MediaPipe（1 人）で各フレームの関節座標を取得。COCO 17 点形式に統一
2. **人物の指定**: ID / クリック位置 / 自動（最も大きく映っている人）。追跡 ID が途中で切り替わった場合は位置と時間の近さでつなぎ直す
3. **前処理**: 低信頼度の関節を補間 → 平滑化 → 30fps にリサンプリング →（必要なら手本を左右反転）→ 体格の正規化（腰の中点を原点、胴の長さを 1）
4. **時間合わせ**
   - 2動画モードで両方に音声があれば、**音声の相互相関で開始位置のずれを求めて揃える**（同じ曲で踊っている前提。自信が無ければ使わない）
   - **DTW** で姿勢の系列を対応づけ、テンポの揺れを吸収 → 1/4 フレーム刻みの局所探索で補正（DTW は 15fps で計算するため）。
     特徴量は各関節の動きを系列内で標準化したもの（振れ幅が小さい・肘が曲がったまま等でも、同じ動きの同じ瞬間が対応するように）
5. **比較**（手本の時間軸で、対応するレビュー対象の値を取り出して比較）
6. **区間ごとの評価**: 1 秒ずつ区切って評価（`window_s_fallback`）。隣り合う同じ指摘はまとめる
7. **出力**: スコア、指摘（優先度 = しきい値に対する倍率 × 継続時間の補正）、比較動画、指摘文

### 部位と評価項目

評価は **部位（足・腰・頭・腕）＋タイミング** ごとにまとめ、部位ごとに重みを付けられます（画面のサイドバー「部位の重み」、
設定ファイルの `weight_legs` などで指定。0 にするとその部位は評価しません）。重みはスコアと「直すところ」の選び方の両方に効きます。

| 部位 | 評価項目 |
|---|---|
| 足 | 股関節・膝の角度、足首の振れ幅・キレ、足幅、着地タイミング |
| 腰（体幹） | 腰の高さ（**各自の普段の高さからの沈み込み**で比べる。体型の違いを除くため）、上体の傾き、体の向き、移動量 |
| 頭（首の動きを含む） | 頭の傾き（両耳の線と両肩の線の角度）、顔の向き（両耳の間の鼻の位置）、横スライド、上下（うなずき・見上げ）。顔が見えにくいコマ・背中向きは評価しない |
| 腕 | 肘・肩の角度、手首の振れ幅・キレ |
| タイミング（全身） | DTW の対応から「同じ動きを手本より何 ms 早く/遅くしているか」。同期済み（同じ動画・音声同期）なら絶対値、そうでなければ全体のテンポ差を除いた局所的なズレ |

指摘には「観点」も付きます: 身体の形（角度・高さ・傾き・向き・足幅・頭）／動きの大きさ（振れ幅・移動量）／キレ（加速・減速の鋭さ）／タイミング（手本との時間差・着地）。
リズム（拍とのズレ）は廃止しました（遅れ・走りは「手本とのタイミング」で評価。拍とのズレの数値は `report.json` の `rhythm` に参考として残しています）。

**スコア**: 1 秒ずつの区間ごとに、各項目のズレが「指摘のしきい値の何倍か」を求めて点数にし、部位ごとに平均します。
しきい値の半分以下 = 100 点、しきい値ちょうど = 80 点、2 倍 = 50 点、3 倍 = 25 点、6 倍以上 = 0 点。
指摘が出ない程度のズレなら 80 点以上になり、しきい値を緩めればスコアも上がります。1 コマごとの骨格推定のぶれは区間の平均で除きます。
総合スコアは部位スコアの重み付き平均です。

**直すところ**: 指摘をすべて見せるのではなく、ズレの大きい瞬間を最大 3 つ（`top_moments`）選び、その前後 1.5 秒（`moment_clip_s`）を
お手本と自分を並べた短い動画（0.5 倍速のスロー, `moment_clip_speed`）にして、ズレている部位に印を付けて見せます（`moment_1.mp4` など。
画面では左に動画・右に修正点。report.md 用に静止画 `moment_1.jpg` も作ります）。一番大きいズレを重く見て、毎エイト同じ癖が出る場合は 1 か所だけ見せて
「同じズレが他に何か所」を添えます。すべての指摘は `findings.csv` と画面の「詳細」で見られます。

- **左右反転**: `--mirror` で手本を反転（x 座標を反転して左右の関節を入れ替え）。部位名はレビュー対象から見た左右になります
- **体格差**: すべて胴の長さで正規化した座標・関節角度で比較します
- **カウント表示（既定では使わない）**: 拍の自動検出は曲によってテンポが倍・半分になったり「1」の位置がずれたりして、
  カウント表示が間違うことが多かったため、既定では場所を時刻で示します。コマンドで `--bpm`（+ `--first-one`）を指定するか
  `--beats`（自動検出）を付けると「2エイト目の3〜4カウント」形式になり、評価も 2 カウントずつ区切ります

---

## 精度調整

実際の動画での調整が一番時間のかかる部分なので、しきい値はすべて `dance_compare/core/config.py` の
`AnalysisConfig` に集めてあり、TOML（`tuning.example.toml` を参照）や UI のスライダー（基本 4 項目 +「高度な設定」）で変えられます。
**どのしきい値も、大きくするほど判定が緩くなり、スコアも上がります。**

調整のヒント:
- 指摘が多すぎる → `angle_threshold_deg` を上げる、`window_counts` を 4 にする
- 振付が「止め」中心なら `accent_mode = "hit"` の方が拍との対応が良いことがある
- 音声同期が誤る（拍 1 つ分ずれる等）→ `--no-audio-sync` か、`audio_sync_min_uniqueness` を上げる
- `report.json` の `windows` に区間ごとの全数値、`timeline` に時系列が入っているので、指摘されなかった箇所の値も確認できます

---

## 構成と、将来の移植に向けた方針

```
dance_compare/
  core/       解析ロジック（numpy のみに依存。I/O・UI・推論エンジンなし）
    skeleton.py   関節の定義（COCO 17 点）, PoseSequence
    preprocess.py 補間・平滑化・左右反転・正規化
    features.py   関節角度・向き・高さ・速度など
    dtw.py        DTW
    rhythm.py     アクセント検出・拍とのズレ・着地検出
    counts.py     拍 → エイト/カウント
    analyzer.py   比較の本体 → Report
    report.py     Report / Finding（UI と言語化はこれだけを見る）
    config.py     しきい値
  pose/       骨格推定バックエンド（yolo / mediapipe）、人物の選択・追跡のつなぎ、キャッシュ
  media/      動画の読み込み、音声の抽出・拍検出・同期
  feedback/   言語化（Narrator を差し替え: rule / gemini / 将来の別 AI・端末内モデル）
  render/     比較動画
  pipeline.py 上記をつなぐ高レベル API
  validation.py 精度検証（ケースファイルの読み込み・照合・結果出力）
  synth.py    合成ダンサー（テスト・検証セット用）
  cli.py      コマンドライン
app.py        Streamlit UI
```

- **骨格推定を端末側へ**: `core` は関節座標の時系列（`PoseSequence`、JSON 化可能）だけを入力にします。
  ブラウザ（MediaPipe Tasks for Web など）やスマホで推定した座標を送れば、サーバー側は `core.compare()` を呼ぶだけです。
  `core` は numpy だけに依存する素直な数値計算なので、TypeScript / Swift / Kotlin への移植も可能な規模にしています
- **言語化の差し替え**: `feedback.base.Narrator`（`narrate(report) -> str`）を実装すれば追加できます
- **API キー**: Gemini は環境変数 `GEMINI_API_KEY`（モデルは `GEMINI_MODEL`, 既定 `gemini-flash-latest`）。
  送るのは動画ではなく数値の要約テキストのみ。無料枠では送信内容が Google の製品改善に使われる可能性があります

## 未決事項について（今回の決定）

- **ローカル UI**: Streamlit（簡易 Web UI）＋ コマンドラインの両方を用意しました
- **1動画モードの人物検出**: Ultralytics YOLO11-pose + ByteTrack を採用（追跡 ID が切れた場合のつなぎ直しを自前で実装）。
  ⚠️ Ultralytics は **AGPL-3.0** です。自分用なら問題ありませんが、Web サービスとして公開する場合はソース公開義務か商用ライセンスが必要です。
  公開時の候補: RTMPose / RTMO（Apache-2.0, `rtmlib` で ONNX 実行可）。`pose/base.py` の `PoseBackend` を実装すれば差し替えられます

## 既知の制約

- 2D の骨格なので、カメラに対して前後方向の動き・体の向き（左右どちらに回ったか）は正確に測れません。手本と同じくらいの角度から撮影するのが理想です
- 足先の点がない（COCO 17 点）ため、ステップは足首で見ています
- 動画の先頭・末尾 0.5 秒にかかる区間はタイミングを、途中で切れた区間は振れ幅・キレを評価しません（端の影響で誤検出しやすいため）
- 音声同期は「同じ曲で踊っている」場合のみ。一定のリズムが続く曲では拍 1 つ分ずれる可能性があるため、確信度が低いときは使いません

## テスト

```bash
pytest
```

合成した棒人間（`dance_compare/synth.py`）に、遅れ・肘の曲げ・振れ幅・キレ・テンポ差・左右反転などを加えて、
期待どおりに検出されるかを確かめています。

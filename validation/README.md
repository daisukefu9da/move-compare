# 精度検証（テスト動画と期待される検出結果）

「答えがわかっているズレ」を入れたテスト動画で、意図したズレが **正しい箇所（カウント）・正しい部位・正しい観点** で、
**想定した大きさ** で検出されるかを一括で確認する仕組みです。

```bash
# 合成データ（棒人間）の見本一式を作って実行してみる
python -m dance_compare make-validation-set --out validation/synth
python -m dance_compare validate validation/synth/cases.toml --out validation_out

# 実際のテスト動画で
cp validation/cases.example.toml validation/cases.toml   # 動画パスと期待値を編集
python -m dance_compare validate validation/cases.toml --out validation_out
python -m dance_compare validate validation/cases.toml --case elbow_bent --video   # 1 ケースだけ, 比較動画付き
```

全ケース合格なら終了コード 0、不合格があれば 1 を返します。骨格推定はキャッシュされるので、
しきい値（`config = "tuning.toml"`）を変えての再検証はすぐ終わります。

## 出力

| ファイル | 内容 |
|---|---|
| `validation_out/validation.md` | 全ケースの合否一覧と、期待ごとの結果・実際の検出・想定外の検出 |
| `validation_out/validation.json` | 同じ内容の機械可読版 |
| `validation_out/<ケース名>/findings.csv` | 検出一覧（カウント・時刻・観点・項目・部位・向き・ズレ量・単位・優先度）。Excel で開ける |
| `validation_out/<ケース名>/report.md` | 通常の解析結果（末尾に検出一覧の表） |

期待ごとの結果:
- **OK**: 条件に合う検出があった
- **値が範囲外**: 観点・部位・場所は合っているが、ズレ量が `min`〜`max` の外
- **場所違い**: 観点・部位は合っているが、別のカウントで検出された
- **未検出**: 該当する観点・部位の検出が無い

「想定外の検出」は、どの `expect` にも当てはまらない検出です（誤検出の候補）。合否には影響しませんが、
`fail_on_unexpected = true` をケースに書くと不合格扱いにできます。

## ケースファイルの書き方（TOML）

```toml
[settings]                 # 全ケース共通（ケース側で上書き可）
count_tolerance = 1        # カウント範囲の前後に許す余裕（カウント数）
first_one = 2.0            # 手本動画で「1エイト目の1」の時刻 [秒]

[[case]]
name = "elbow_bent"
ref = "videos/ref.mp4"     # このファイルからの相対パス
user = "videos/elbow_bent.mp4"
  [[case.expect]]
  aspect = "shape"
  part = "right_elbow"
  direction = "smaller"
  start = "2-1"            # 「エイト-カウント」。裏拍は "2-3&"
  end = "2-8"              # このカウントを含む
  max = -20
  [[case.forbid]]
  aspect = "timing"
```

### ケースに書ける項目

| 項目 | 内容 |
|---|---|
| `ref`, `user` | 2動画モードの動画 |
| `mode = "single"`, `video` | 1動画モード。`ref_id`/`user_id` または `ref_point`/`user_point`（`[x, y]`）と `frame` |
| `ref_pose`, `user_pose` | 骨格 JSON（解析結果の `ref_pose.json` など）を使い、骨格推定を省略 |
| `mirror`, `bpm`, `first_one`, `no_beats`, `audio_sync` | 解析オプション（CLI と同じ意味） |
| `ref_start`, `ref_end`, `user_start`, `user_end` / `start`, `end` | 区間の指定 [秒] |
| `config` | 解析パラメータの TOML |
| `count_tolerance`, `fail_on_unexpected` | 照合の設定 |

### `expect` / `forbid` に書ける項目

| 項目 | 内容 |
|---|---|
| `aspect`（必須） | 観点（下表） |
| `kind` | 項目（下表）。省略すると観点内のすべて |
| `part` | 部位。リストならどれか 1 つで可 |
| `direction` | ズレの向き |
| `start`, `end` | カウント範囲（`"2-1"` 形式）。省略すると「全体の傾向」として照合 |
| `start_s`, `end_s` | 秒で範囲を指定する場合（手本動画の時刻） |
| `min`, `max` | ズレ量の範囲（符号付き。単位は下表） |

| aspect | kind | part | direction | ズレ量の単位 |
|---|---|---|---|---|
| `shape`（身体の形） | `angle` | `left/right_elbow`, `_shoulder`, `_hip`, `_knee` | `smaller`（曲がっている・腕/脚が下がっている）/ `larger` | 度（レビュー対象 − 手本） |
| `timing`（手本とのタイミング） | `early_late` | - | `late` / `early` | ms（正 = 遅い） |
| `rhythm`（リズム） | **廃止**（検出されない） | | | |
| `steps`（ステップ） | `landing` | `left/right_ankle`（片足だけの場合） | `late` / `early` | ms |
| | `stance` | - | `wider` / `narrower` | 胴の長さ比 |
| | `travel` | - | `less` / `more` | 胴の長さ比 |
| `amplitude`（動きの大きさ） | `amplitude` | `left/right_wrist`, `left/right_ankle` | `smaller` / `larger` | 手本に対する比（0.7 = 70%） |
| `sharpness`（キレ） | `sharpness` | 同上 | `weaker` | 手本に対する比 |
| `level`（レベル・向き） | `level` | - | `higher` / `lower` | 胴の長さ比 |
| | `facing` | - | `more_turned` / `less_turned` | 度 |
| | `lean` | - | `right` / `left` | 度 |

## テスト動画を撮るときの注意

- 手本と同じ位置・角度から撮ると、2D 骨格の誤差が揃って比較しやすくなります
- 拍のカウントがずれないよう、`first_one`（手本動画で「1エイト目の1」の時刻）を指定してください。
  自動検出では最初の拍を取りこぼして 1 拍ずれることがあります
- **全体を半拍遅らせた場合**は、手本との時間差（`timing` / `early_late`）と、着地のズレ（`steps` / `landing`）で検出されます
- 振れ幅を抑えると、腕が上がりきらない分「身体の形（肩の角度）」の差も出ます。これは誤検出ではないので、
  必要なら `expect` に加えるか、想定外の検出として確認してください
- 肘を曲げると手先の振れ幅も小さくなるので、「動きの大きさ（手）」が想定外の検出として出ることがあります

## 合成データの見本について

`make-validation-set` で作られる 4 ケース（肘を曲げたまま / 半拍遅れ / 3エイト目の着地だけ遅らせる / 振れ幅 60%）は、
仕組みの動作確認と、解析ロジックを変更したときの回帰テスト用です（`tests/test_validation.py` でも実行しています）。
棒人間なので骨格推定の誤差はなく、実際の動画での精度はここより下がります。

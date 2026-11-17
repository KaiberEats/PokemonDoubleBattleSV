# 学習時間の計測

通常の train_team.py 実行で、PPO更新・保存ごとに `[TIMING]` の表を表示します。
計測は実際に関数を呼ぶスレッドで行い、スレッド間で集計します。

| name | 内容 |
|---|---|
| collection_wall | 前回の更新後、対戦収集と全担当の待ち合わせにかかった実時間 |
| history_record | PPOプレイヤーの公開イベント履歴の記録 |
| observation | PPOの観測ベクトル生成（選出と対戦を含む） |
| action_mask | 対戦中の合法行動マスク生成 |
| preview_mask | 選出候補のマスク生成 |
| inference | 対戦中のモデル推論と行動抽選 |
| preview_inference | 選出時のモデル推論と行動抽選 |
| action_to_order | 行動IDから送信する行動への変換 |
| buffer_merge | プレイヤー別バッファの集約 |
| ppo_update | モデル更新（GPU使用時は完了を同期して計測） |
| model_save | 更新後のモデル保存 |

calls は回数、wall_s は呼び出し時間の合計秒数、thread_cpu_s は呼出元スレッドの
CPU時間の合計、avg_ms/max_ms は1呼び出しの平均/最大ミリ秒です。
thread_cpu_s はPyTorch内部の別スレッドやShowdownプロセスのCPU時間を含みません。
表は前の表示からの区間集計です。
collection_wall は内訳の処理・通信・相手側の処理・待ち時間を含み、
各項目のwall_sとの合計は全体時間になりません。

SSH先で出力を保存する例：

```bash
python train_team.py --parallel-battles 5 --battles-per-opponent 100 2>&1 | tee logs/speed_5.log
```

同じ初期モデルと総対戦数で並列数を変更して比較します。
observation/action_mask/history_record が支配的ならPython前処理を、
inference が支配的なら推論を、ppo_update が支配的なら更新処理を優先して改善します。
計測のオーバーヘッドを除いて測りたい場合：

```bash
DOUBLEBATTLE_PROFILE=0 python train_team.py --parallel-battles 5
```

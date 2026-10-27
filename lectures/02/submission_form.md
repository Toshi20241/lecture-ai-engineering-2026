# Day2 提出フォームと手順

## Q1

実装｜Implementation

## Q2

https://github.com/Toshi20241/lecture-ai-engineering-2026

## Q3（139文字）

製造業のDX相談を題材に、架空のFAQ5項目を使ってLoRA学習を行いました。学習前後の内容正解率は57.1%から85.7%に改善しました。FAQにない質問では、別のFAQの回答を返す誤りが残りました。今回の実験を通じて、正解率が上がっても回答内容の確認が必要だと分かりました。

## GitHubへの配置

1. `day2_submission_simple.zip`をダウンロードして解凍する。
2. `day2_submission_simple`フォルダを開く。中にある`lectures`と`notebooks`がアップロードする2つのフォルダ。
3. 自分のリポジトリを開き、ブランチ選択欄で`main`を選ぶ。そこから`lec02`を作成する。作成済みなら`lec02`を選ぶ。
4. リポジトリの最上位で`Add file`→`Upload files`を選び、2つのフォルダをドラッグする。
5. `lectures/02/answer.md`と`notebooks/day2_dx_faq_lora.ipynb`が一覧にあることを確認する。
6. コミットの説明を`Day2 実装課題`とし、`lec02`へ保存する。
7. [比較画面](https://github.com/Toshi20241/lecture-ai-engineering-2026/compare/main...lec02?expand=1)を開き、自分のリポジトリ内の`lec02`→`main`でPRを作る。タイトルは`Day2 実装課題`。本文は「独自FAQを使ったLoRA学習の結果と考察を提出します。」。
8. PR作成後、QuriにQ1〜Q3を入力し、提出完了画面を確認する。

課題で求められるGitHub上の到達点はPR作成。マージは要件に含まれていない。
学習結果は取得済みで、再学習は必要ない。

## ファイルの内容

- `answer.md`：演習・宿題の解答。
- `results/day2_dx_faq/`：実測結果と学習前後の全回答。元のログを変更せず保存。
- `../../notebooks/day2_dx_faq_lora.ipynb`：使用したコードとデータを含むノートブック。

ノートブックは再実行用のテンプレートで、今回の実測値は結果JSONに保存している。

# GitHub公開手順

推奨repository名は `union-selection-sieve` です.

1. GitHubで新規repositoryを作成します. README, LICENSE, `.gitignore`はこのpackageに含まれているため, GitHub側では自動生成しません.
2. 本packageを展開し, 展開後の`union-selection-sieve`ディレクトリをrepository rootとして使用します.
3. 最初のcommit前に`CITATION.cff`, `.zenodo.json`, `pyproject.toml`内のrepository URLを確認します.
4. ローカルで次を実行します.

```bash
python -m pip install -e ".[dev]"
pytest
union-selection-demo --outdir outputs/demo
python scripts/run_from_config.py configs/smoke.json
```

5. 成功後にcommitしてpushします.

```bash
git init
git add .
git commit -m "Initial release of the union-selection sieve estimator"
git branch -M main
git remote add origin https://github.com/TsutomuTakeuchiTTT/union-selection-sieve.git
git push -u origin main
```

6. GitHub上でActionsが成功することを確認します.
7. 初回公開版には`v1.0.0` tagを付けます.
8. Zenodo連携を行う場合はGitHub release作成後にDOIを取得し, DOIを`CITATION.cff`, `.zenodo.json`, README, 論文のCode availabilityへ反映します.

旧repositoryとは方法が異なるため, 旧コードをこのrepositoryへ混在させないことが重要です.

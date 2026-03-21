## プラットフォーム注意事項
- ファイル操作はPowerShellコマンド（Move-Item, Copy-Item）を優先すること
- Git Bashでのrobocopyはパス変換問題で失敗するため避ける
- シンボリックリンクはNTFSジャンクション（mklink /J）を使用してください
- シェルはデフォルトでpwsh を利用してください
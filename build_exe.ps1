<#
  build_exe.ps1 — allin1 を PyInstaller でパッケージングするスクリプト
  
  使い方:
    .\build_exe.ps1           # 通常ビルド（--onedir）
    .\build_exe.ps1 -OneFile  # 単一EXEビルド（起動が数十秒遅くなる場合あり）
    .\build_exe.ps1 -Clean    # ビルド成果物を削除してから再ビルド
#>

param(
    [switch]$OneFile,
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot

Write-Host "=== allin1 EXE ビルド ===" -ForegroundColor Cyan

# ─── クリーンアップ ───────────────────────────────────────────────────────
if ($Clean) {
    Write-Host "-> ビルド成果物を削除中..." -ForegroundColor Yellow
    @('dist', 'build') | ForEach-Object {
        $p = Join-Path $ScriptDir $_
        if (Test-Path $p) { Remove-Item $p -Recurse -Force }
    }
}

# ─── PyInstaller のインストール確認 ──────────────────────────────────────
Write-Host "-> PyInstaller の確認..." -ForegroundColor Yellow
$pyinstallerCmd = $null

# uv 経由で動作中の仮想環境を優先する
if (Get-Command uv -ErrorAction SilentlyContinue) {
    # uv run で pyinstaller を解決できるか試す
    $piVersion = uv run python -c "import PyInstaller; print(PyInstaller.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pyinstallerCmd = 'uv run pyinstaller'
        Write-Host "   uv 仮想環境の PyInstaller $piVersion を使用します" -ForegroundColor Green
    } else {
        Write-Host "   PyInstaller が見つかりません。インストールします..." -ForegroundColor Yellow
        uv add --dev pyinstaller
        $pyinstallerCmd = 'uv run pyinstaller'
    }
} elseif (Get-Command pyinstaller -ErrorAction SilentlyContinue) {
    $pyinstallerCmd = 'pyinstaller'
} else {
    Write-Error "PyInstaller が見つかりません。'pip install pyinstaller' または 'uv add --dev pyinstaller' でインストールしてください。"
    exit 1
}

# ─── ビルド実行 ───────────────────────────────────────────────────────────
Push-Location $ScriptDir

try {
    if ($OneFile) {
        # --onefile: 単一 EXE（起動時に %TEMP% へ展開するため初回が遅い）
        Write-Host "-> --onefile モードでビルドを開始します..." -ForegroundColor Yellow
        Write-Host "   ※ 注意: PyTorch 等の DLL が巨大なため、起動に数十秒かかる場合があります" -ForegroundColor Yellow

        # spec を一時的に書き換えてオンファイルビルドを実行
        $specContent = Get-Content allin1.spec -Raw
        $specContentOnefile = $specContent `
            -replace "exclude_binaries=True",   "exclude_binaries=False" `
            -replace "# --onedir: バイナリは別フォルダへ", "# --onefile モード"

        # COLLECT ブロックを除いた一時 spec を生成
        $specContentOnefile = $specContentOnefile -replace "(?s)coll\s*=\s*COLLECT\(.*?\)", ""

        # EXE の scripts 引数に a.binaries / a.datas を追加
        $specContentOnefile = $specContentOnefile -replace `
            "exe = EXE\(\s*pyz,\s*a\.scripts,\s*\[\],", `
            "exe = EXE(`n    pyz,`n    a.scripts,`n    a.binaries,`n    a.datas,"

        $tmpSpec = Join-Path $ScriptDir 'allin1_onefile.spec'
        $specContentOnefile | Set-Content $tmpSpec -Encoding UTF8

        Invoke-Expression "$pyinstallerCmd allin1_onefile.spec --clean"
        Remove-Item $tmpSpec -Force
    } else {
        # --onedir（推奨）: dist/allin1/ フォルダ + allin1.exe
        Write-Host "-> --onedir モードでビルドを開始します..." -ForegroundColor Yellow
        Invoke-Expression "$pyinstallerCmd allin1.spec --clean"
    }

    Write-Host ""
    Write-Host "=== ビルド完了 ===" -ForegroundColor Green

    if ($OneFile) {
        $exePath = Join-Path $ScriptDir 'dist\allin1.exe'
        Write-Host "生成ファイル: $exePath" -ForegroundColor Cyan
    } else {
        $dirPath  = Join-Path $ScriptDir 'dist\allin1'
        $exePath  = Join-Path $dirPath 'allin1.exe'
        Write-Host "生成フォルダ: $dirPath" -ForegroundColor Cyan
        Write-Host "実行ファイル: $exePath" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "配布方法: dist\allin1\ フォルダごと ZIP して配布してください。" -ForegroundColor Yellow
        Write-Host "  Compress-Archive -Path '$dirPath' -DestinationPath 'allin1.zip'"
    }

    Write-Host ""
    Write-Host "使用例:" -ForegroundColor Cyan
    Write-Host "  $exePath 'C:\音楽\sample.mp3' --overwrite --keep-byproducts"

} finally {
    Pop-Location
}

#!/bin/zsh
# このファイルがあるフォルダへ移動し、PixiでROS 2環境を選んで実行します。
set -eu
cd "$(dirname "$0")"
# 通常はインストール済みPixiを使います。この作品の制作環境も利用できます。
if command -v pixi >/dev/null 2>&1; then
    pixi_bin=$(command -v pixi)
elif [[ -x ../../work/bin/pixi ]]; then
    pixi_bin=../../work/bin/pixi
else
    echo "Pixiが見つかりません。READMEの『動かす準備』を確認してください。"
    exit 1
fi
# ROSのログは作品フォルダ内へ保存します。
export ROS_LOG_DIR="$PWD/logs"
# frozenは、記録済みのバージョンを使って環境を再現する指定です。
exec "$pixi_bin" run --frozen export

#!/bin/bash

# 设置项目路径（可选：你也可以把这行注释掉）
cd /home/astwea/MyDogTask/Mydog

# 获取当前时间作为版本号
version="v$(date +'%Y-%m-%d_%H-%M')"
message="Auto commit: $version"

# 添加所有更改
git add .

# 提交（如无更改会报错，但不影响）
git commit -m "$message"

# 打 tag（如果 tag 名已存在则跳过）
if ! git rev-parse "$version" >/dev/null 2>&1; then
    git tag "$version"
fi

# 推送提交和 tag
git push origin main
git push origin "$version"

echo "✅ 已成功提交并推送：$version"


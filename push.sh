#!/bin/bash

# 设置项目路径（可选：你也可以把这行注释掉）
cd /home/astwea/MyDogTask/Mydog

# 获取当前时间作为版本号
version="v$(date +'%Y-%m-%d_%H-%M')"

# 让用户输入提交信息（更新日志）
read -p "请输入更新日志（提交说明）: " user_message

# 如果用户没有输入，则使用默认信息
if [ -z "$user_message" ]; then
    user_message="Auto commit: $version"
fi

# 添加所有更改
git add .

# 提交更改
git commit -m "$user_message"

# 打 tag（如果 tag 名已存在则跳过）
if ! git rev-parse "$version" >/dev/null 2>&1; then
    git tag "$version"
fi

# 推送提交和 tag
git push origin main
git push origin "$version"

echo "✅ 已成功提交并推送：$version"


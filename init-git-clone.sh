#!/bin/bash



# shellcheck disable=SC2164
echo "반드시 Github 개인 계정에 등록후 사용하세요"
cd /data/ephemeral/home
git clone git@github.com:boostcampaitech7/level2-competitiveds-recsys-06.git
git checkout main
git pull origin main


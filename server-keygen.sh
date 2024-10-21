#!/bin/bash



ssh server4 << 'ENDSSH'
apt-get update -y && apt-get upgrade -y
apt-get  install docker.io curl -y
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
docker-compose --version

git config --global user.name "홍길동(이름)"
git config --global user.email "test@example.com"

ssh-keygen -t ed25519 -C "xenx96@naver.com" -N "" -f ~/.ssh/id_ed25519 -y

echo "SSH 키가 생성되었습니다. 다음 명령어로 공개 키를 복사 후 Git 개인 SSH에 등록하세요(Auth&Sign 둘다):"
cat /root/.ssh/id_ed25519.pub
ENDSSH

echo "모든 작업이 완료되었습니다."
read


source .env

ssh server$server_no << 'ENDSSH'
if pgrep -f "python app.py" > /dev/null
then
    echo "app.py가 실행 중입니다. 종료를 시작합니다."

    # 실행 중인 app.py 프로세스의 PID를 찾아서 강제 종료
    pkill -f "python app.py"

    if [ $? -eq 0 ]; then
        echo "app.py가 성공적으로 종료되었습니다."
    else
        echo "app.py 종료에 실패했습니다."
    fi
else
    echo "app.py가 실행 중이 아닙니다."
fi
ENDSSH
read
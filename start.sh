sudo kill -9 $(sudo ps e | grep server.py | awk '{print $1}')
git pull
sudo /etc/init.d/apache2 stop
nohup sudo FLASK_APP=server.py python3 -m flask run --host=0.0.0.0 --port=80 &
                                                                                    
import socket

HOST = '192.168.56.1'
PORT = 65433

FILLER = '          '

import hbase
import pickle

zk = '127.0.0.1'

HEADERSIZE = 10

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
        except:
            print("Can't connect")
            exit()
                
        with hbase.ConnectionPool(zk).connect() as conn:
            table = conn['default']['data']
            for row in table.scan():
                data = {'features': row['cf:features'], 'labels': row['cf:labels']}
                msg = pickle.dumps(data)
                msg_len = str(len(msg))
                header = msg_len + FILLER[len(msg_len):]
                msg = str.encode(header) + msg
                s.send(msg)
                responce = s.recv(1024)
                print(responce)     
    exit()




	


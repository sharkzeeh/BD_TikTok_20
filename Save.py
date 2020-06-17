import socket

HOST = '192.168.56.2'
PORT = 65432

import hbase
import pickle

zk = '127.0.0.1'

HEADERSIZE = 10

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, PORT))
            s.listen(5)
        except:
            print("Can't connect")
            exit()

        full_msg = b''
        new_msg = True
        while True:
            socket, address = s.accept()
            while True:
                msg = socket.recv(51200)
                if msg == b'':
                    socket.close()
                    exit()
                if new_msg:
                    print("new msg len:", msg[:HEADERSIZE])
                    msglen = int(msg[:HEADERSIZE])
                    new_msg = False

                full_msg += msg

                if msglen == len(full_msg) - HEADERSIZE:
                    print("full msg recvd")
                    data = pickle.loads(full_msg[HEADERSIZE:])
                    print(data)
                    new_msg = True
                    if not data:
                        break
                    
                    full_msg = b''
                    
                    with hbase.ConnectionPool(zk).connect() as conn:
                        table = conn['default']['data']
                        table.put(hbase.Row(
                            str(data['id']), {
                                'cf:labels':pickle.dumps(data['labels']), 
                                'cf:features':pickle.dumps(data['features'])
                            }
                        ))
                        table.get(str(data['id']))
                        conn.close()
                    socket.sendall(b"Received message")
                 
    exit()




	


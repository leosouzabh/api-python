import os
import sqlite3


def connect():
    db_filename = 'todo.db'
    """ Make connection to an SQLite database file """
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    return conn, c

def create():
    
    conn,c = connect()
    
    print ('Creating schema')
                
    conn.execute("""
    create table parametro (
        id           integer primary key autoincrement not null,
        tolerancia   integer,
        perc_tamanho integer,
        pix_branco   integer
    );
    """)


    print ('Inserting initial data')            
    conn.execute("""
    insert into parametro (tolerancia, perc_tamanho, pix_branco)
    values (9, 100, 100)
    """)

    print ('data inserted')            
    conn.commit()
              
def select():
    conn,c = connect()
    c.execute('SELECT * FROM parametro')    
    all_rows = c.fetchall()
    conn.close()

    return all_rows

if __name__ == '__main__':
    create()  
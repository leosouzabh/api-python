class AppException(Exception):
    pass

class QtdeAssinaturasException(Exception):
     def __init__(self, message, identificador):
        super(QtdeAssinaturasException, self).__init__(message)
        self.identificador = identificador
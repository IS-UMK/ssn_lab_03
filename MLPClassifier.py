
def softmax(z):
    # stabilna numerycznie funkcja softmax(x)
    e = np.exp(z - z.max())
    return e/np.sum(e)


def log_softmax(x):
    # stablilna numerycznie operacja log(softmax(x))
    x_max = np.max(x)
    return x - x_max - np.log(np.sum(np.exp(x - x_max)))


class MLPClassifier:
    
    def __init__(self, eta=0.1, n_epochs=100):
        
        self.n_epochs = n_epochs   # ilość epok
        self.eta = eta             # stała uczenia

    def accuracy(self, X, y):
        # poprawnośc klasyfikacji 
        return (self.predict(X) == y).mean()

        
    def init(self, X, y):
        
        # TODO utworzenie i zainicjowanie wag 
        return self

   
    def feedforward(self, x):
        
        # TODO: propagacja sygnału sygnału 
        # zwraca wyjscie sieci
        pass

    
    def backprop(self, y_one):
        
        # TODO: propagacja wsteczna sygnału błedu 
        return self

    
    def update(self, x):
        
        # TODO: aktualizacja wag
        return self

    
    def predict(self, X):

        # TODO: zwraca etykiety klas 
        pass    


    def fit(self, X, y):
        
        y_one = OneHotEncoder().fit_transform(y[:, np.newaxis]).toarray()
        self.init(X, y_one)
        
        self.accuracies = []
        self.losses = []
        ind = np.arange(X.shape[0])

        for i_epoch in range(self.n_epochs):
            
            loss = 0
            for i in np.random.permutation(X.shape[0]):
                output = self.feedforward(X[i, :])
                self.backprop(y_one[i])
                self.update(X[i, :])
                
                # loss -= np.log(output[y[i]])
                loss -= log_softmax(output)[y[i]]

            self.losses.append(loss/X.shape[0])
            self.accuracies.append(self.accuracy(X, y))

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from numpy import array
import pickle

#IMPORTING THIS FILE
#   import sys
#   import os
#   sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#   import My_ML_Functions.Deep_Learning as dl



class Classification(object):

    def hypothesis(self , x, th, th0, activation):
        return activation(np.dot(x, th) + th0)

    def cross_entropy_loss(self, guess, y):
        guess = np.clip(guess , 1e-15 , 1-(1e-15))
        return  -( ( y * np.log(guess) ) + ( (1-y) * np.log(1-guess) ) )
    def perceptron_loss(self, guess, y):
        return np.maximum(0 , -y*guess)
    def hinge_loss(self, guess, y):
        return np.maximum(0 , 1-(y*guess))
    def no_loss(self , guess, y):
        return 0
    
    def l1_regulizer(self, W):
        return np.abs(W)
    def dl1(self, W):
        return np.sign(W)
    def l2_regulizer(self, W):
        return W**2
    def dl2(self, W):
        return 2*W
    def no_regulization(self , W):
        return 0
    
    def sigmoid(self , W):
        W = np.clip(W , -500, 500)
        return 1/(1+np.exp(-W))
    

    def __init__(self, X , Y , model="perceptron" , loss="no loss" , regulizer=None):
        self.x = np.array(X , ndmin=2)
        self.y = np.array(Y , ndmin=2).reshape(-1,1)
        self.losses = []

        self.model = model
        if loss.lower() == "perceptron":
            self.loss = self.perceptron_loss
        elif loss.lower() == "hinge":
            self.loss = self.hinge_loss
        elif loss.lower() in ["cross entropy","log loss", "ce" , "ll", "nll", "negetive log liklihood"]:
            self.loss = self.cross_entropy_loss
        else:
            self.loss = self.no_loss

        if regulizer is None:
            self.regulizer = self.no_regulization
            self.dR = self.no_regulization
        elif regulizer.lower() == "l1":
            self.regulizer = self.l1_regulizer
            self.dR = self.dl1
        elif regulizer.lower() == "l2":
            self.dR = self.dl2
            self.regulizer = self.l2_regulizer

    def perceptron(self , T):

        X , Y = self.x , self.y
        samples , features = X.shape
        th , th0 =  np.zeros((features,1)) , 0


        for t in range(T):
            for i in range(samples):
                xi , yi = X[i:i+1 , :] , Y[i , 0]
    
                if yi*(np.dot(xi,th) + th0) <= 0:
                    th += (yi*xi).T
                    th0 += yi
            guess = self.hypothesis( X, th, th0, np.sign )
            self.losses.append(np.mean(self.loss(guess , Y)))

        
        self.th , self.th0 = th , th0

    def gradient_descend(self , F, dF ,iterations , lr, epsilon):
        features = self.x.shape[1]

        th , th0 = np.zeros(features) , 0
        TH = np.append(th , th0)

        for t in range(1,iterations+1):
            grad = dF(TH)
            TH -= lr(t) * grad
            if np.linalg.norm(grad) <= epsilon:
                break
            guess = self.hypothesis( self.x, TH[:-1].reshape(-1,1), TH[-1], self.sigmoid )
            self.losses.append(np.mean(self.loss(guess , self.y)) + np.sum(self.regulizer(TH[:-1].reshape(-1,1))))

        
        self.th , self.th0 = TH[:-1].reshape(-1,1) , TH[-1]

    def optimization(self , lr , iterations=10000 , epsilon=0.0001):

        def J(TH):
            th , th0 = TH[:-1].reshape(-1,1) , TH[-1]
            return (1/len(self.x))*np.sum(self.loss( self.hypothesis( self.x , th, th0 , self.sigmoid) , self.y)+self.regulizer(th))
        
        def dJ(TH):
            th , th0 = TH[:-1] , TH[-1]
            N = len(self.x)

            guesses = self.hypothesis(self.x, th, th0, self.sigmoid)

            dL = (1/len(self.x))*self.hypothesis( self.x.T , th, th0 , self.sigmoid)*(1 - self.hypothesis( self.x.T , th, th0 , self.sigmoid) )*self.x.T

            if self.loss == self.cross_entropy_loss:
                guesses = self.hypothesis(self.x, th, th0, self.sigmoid)
                error = guesses-self.y

                d_th = (1 / N) * np.dot(self.x.T, error) + self.dR(th)
                d_th0 = (1 / N) * np.sum(error)
            else:
                d_th , d_th0 = np.zeros_like(th) , 0.0

            return np.append(d_th.flatten(), d_th0)


        self.gradient_descend(J, dJ, iterations, lr, epsilon)

    def Train_classifier(self, Dataset=None, lr=(lambda t: 0.01) , iterations=1000, epsilon=0.001 , epoch=10):
        if Dataset is not None:
            self.x = np.array([i[0] for i in Dataset], ndmin=2) # D = [[[x_array],y]...]
            self.y = np.array([i[1] for i in Dataset], ndmin=2).reshape(-1,1)
        
        if self.model == "perceptron":
            self.model = self.perceptron
            self.model(epoch)

        elif self.model == "optimization":
            self.model == self.optimization
            self.model(lr,iterations,epsilon)

        else:
            return "Invalid Model, please change it to optmization or Perceptron"
        
        return {"loss":self.losses , "model":(self.th , self.th0)}

class Regression(object):

    def __init__(self):
        pass        

#Writing Y is I can self makt it from scratch , and MB if I maybe able to build it from scratch or 
#partially able to make it from scratch , and N if am unable to make it from scratch

class FF_Neural_Network(object):

    #3 Criterias for this function to work
    # 1) Data format - Dataset(both Testing and Training) = [ [x(1) , y(1)] , [x(2) , y(1)] ,..., [x(n) , y(n)] ] where x(i) , y(i) {i in range(1,n+1)} are column vectors
    #    with x.shape = (1,features) and y.shape = (1,classes) for MCC and (1,1) for Regression
    # 2) The layers while defining the class should match the dataset , so layer = [Dataset[i][0].shape[1] , ...hidden_layers... , Dataset[i][1].shape[1]] where i in range(1,n+1)
    # 3) Right activation , lla, loss and lr(t) should be selected based on the Data 

    #If entering X, Y seperatly keep in Mind that both are matrices with
    #Dimention (features , samples) so that indexing is possible

    def weight_initalization(self , n_in , n_out , activation="sigmoid"):   #Y this is just Weights Initalization
        if activation.lower() == "relu":
            return (np.random.randn(n_in , n_out) * np.sqrt(2/n_in))    # Made for ReLU as Activation
        else:
            return (np.random.randn(n_in , n_out) * np.sqrt(1/n_in))    #Made for Normal weight initalization
        
    def __init__(self , layer_size , activation="sigmoid"): #Y this is initalization , W = (output , input) * x = (input , 1) + b=(output , 1)== (output,1)+(output,1) = (output,1) 
        self._activation_ = activation
        self.layers = layer_size    #Returnes the whole Neural Network Architecture
        self.num_layers = len(layer_size)   #Return the Number of Layers of NN
        self.input_layer , self.output_layer , self.hidden_layers = self.layers[0] , self.layers[-1] , self.layers[1:-1]    #Returnes all kinds of layers
        self.weights , self.biases = [] , []
        

        for l in range(self.num_layers-1):
            self.weights.append( self.weight_initalization(self.layers[l+1] , self.layers[l] , activation) )
            self.biases.append( np.zeros((self.layers[l+1] , 1)) )

    def Loss_CE(self , AL , y, Catagorial=True):  #Y ,This is Catagorial/Non Catagorial Loss_Cross_Entropy
        AL = np.clip(AL,1e-12 , 1-1e-12)
        if Catagorial is True:
            return -np.sum(y*np.log(AL))/AL.shape[1]
        else:
            return -np.sum(y*np.log(AL) + (1-y)*np.log(1-AL))/AL.shape[1]

    def Loss_sq(self , AL ,y):  #Y
        return np.mean((AL-y)**2)
    
    #Activation Functions
    def sigmoid(Z): #Y
        return 1/(1+(np.exp(-(np.clip(Z,-500,500)))))
    def tanh(Z):    #Y
        return np.tanh(Z)
    def ReLU(Z):    #Y
        return np.maximum(0 , Z)
    
    #Activation Function Derivatives
    def dSig(Z):    #Y
        return FF_Neural_Network.sigmoid(Z)*(1-FF_Neural_Network.sigmoid(Z))
    def dtanh(Z):   #Y
        return 1-(np.tanh(Z)**2)
    def dReLU(Z):   #Y
        return (Z>0).astype(float)
    def dsame(Z):   #Y
        return np.ones_like(Z)

    #Last Layer Activations
    def softmax(Z): #Y
        shift_Z = Z - np.max(Z, axis=0, keepdims=True)
        return np.exp(shift_Z)/np.sum(np.exp(shift_Z),axis=0,keepdims=True)
    def same(Z):    #Y
        return Z
    
    #learning rate
    def lr_constant(t , k=0.01):    #Y
        return k

    #Main Functions   
    #Forward Pass
    def forward_pass(self, X ,Y=None , activation="sigmoid", lla="softmax", Sloss="cross entropy"):    #Y, activation list, Z list len(activations)=len(Zs)+1
        if activation.lower() == "sigmoid":
            self.activation_f = FF_Neural_Network.sigmoid
        elif activation.lower() == "tanh":
            self.activation_f = FF_Neural_Network.tanh
        else:
            self.activation_f = FF_Neural_Network.ReLU

        if lla.lower() == "softmax":
            self.activation_llf = FF_Neural_Network.softmax
        elif lla.lower() == "linear":
            self.activation_llf = FF_Neural_Network.same
        else:
            self.activation_llf = self.activation_f

        if Sloss.lower() == "cross entropy":
            self._loss = self.Loss_CE
        elif Sloss.lower() in ["squared", "mse", "mean squared error"]:
            self._loss = self.Loss_sq


        self.activations , self.Zs = [X] , []
        for i in range(len(self.weights)):
            A = self.activations[-1]
            Z = self.weights[i]@A + self.biases[i]
            self.Zs.append(Z)

            if i != len(self.weights)-1:
                self.activations.append(self.activation_f(Z))
            else:
                self.activations.append(self.activation_llf(Z))

        if Y is not None:
            return self._loss(self.activations[-1] , Y)
        return self.activations[-1]
    #Backward Pass
    def backward_pass(self, X, Y, loss="cross entropy", activation="sigmoid", lla="softmax", fowardpass=False):  
        self.loss_type_str = loss.lower()

        if fowardpass is False:
            self.forward_pass(X, Y, activation=activation, lla=lla)
        
        f, lf = self.activation_f, self.activation_llf
        
        if loss.lower() == "cross entropy":
            self.loss = self.Loss_CE
        elif loss.lower() in ["squared", "mse", "mean squared error"]:
            self.loss = self.Loss_sq

        AL = self.activations[-1]
        m = X.shape[1]

        deriv_map = {
            FF_Neural_Network.sigmoid: FF_Neural_Network.dSig,
            FF_Neural_Network.tanh: FF_Neural_Network.dtanh,
            FF_Neural_Network.ReLU: FF_Neural_Network.dReLU,
            FF_Neural_Network.same: FF_Neural_Network.dsame
        }

        # Fixed explicit checks for clean gradient derivation
        if self.loss_type_str == "cross entropy" and lla.lower() == "softmax":
            dZ = AL - Y
        elif self.loss_type_str in ["squared", "mse", "mean squared error"] and lla.lower() in ["linear", "same"]:
            dZ = (2/m) * (AL - Y)
        else:
            f_prime = deriv_map.get(lf, FF_Neural_Network.dsame)
            dZ = (2/m) * (AL - Y) * f_prime(self.Zs[-1])

        self.dw , self.db = [] ,[]

        def back_propagate():
            nonlocal dZ
            df = deriv_map.get(self.activation_f, FF_Neural_Network.dsame)

            for i in range(len(self.weights)-1, -1, -1):
                A_i = self.activations[i]
                dW = (1/m)*dZ@A_i.T #Taking avg across all the k(columns) of the matrix , by dW = ((dZ)(A^T))/m this is just chain rule
                dB = (1/m)*np.sum(dZ , axis=1, keepdims=True)   #Derivative of that with dB is just 1, so just summed it all up and took avg
                self.db.append(dB), self.dw.append(dW)  #Appeding the gradients to the self.db , self.dw

                if i > 0:
                    dZ = (self.weights[i].T@dZ)*df(self.Zs[i-1])  #Updating dA/dZ by chain rule, dZ/dA * dA/dZ == W_i * dZ * f'(Z_i)
                else:
                    self.dZ_1 = dZ
            self.dw = self.dw[::-1] #Reversing the order for better usage
            self.db = self.db[::-1] #Reversing the order for better usage
        back_propagate() #Calling the function

    #RUNNING FUNCTIONS USED NORMALLY 
    #Stocastic Batch Training
    def Train_NN(self, Dataset=None, X1=None, Y1=None, k=1, iterations=1, epoch=1, lr=None, lam=0, loss="cross entropy", activation="sigmoid",lla="softmax",interval=100):  #Y
        Start = time.time()
        if lr is None:
            lr = FF_Neural_Network.lr_constant
    
        if (X1 is None) and (Y1 is None) and (Dataset is not None):
            X_all = np.column_stack([item[0] for item in Dataset])  #Stacking X,Y column vise
            Y_all = np.column_stack([item[1] for item in Dataset])
        elif (X1 is not None) and (Y1 is not None):
            X_all = X1
            Y_all = Y1
        else:
            print("WARNING : You must enter (X and Y) or the Datast, is none is entered the training will fail")
            return None
                
        num_samples = X_all.shape[1]    #Our batch size
        self.losses_over_t = []

        #print("Data Initialization Done")

        for t in range(1,iterations+1):
            
            if t == 1:
                #print("Loop Begun")
                pass
            
            indices = np.random.choice(num_samples , size=k , replace=False)    #Taking k random numbers in rang 0,num_saples
            X = X_all[:, indices]   #Using those samples to get our X , Y 
            Y = Y_all[:, indices]
            
            if t == 1:
                #print("Epoch loop begun")
                pass

            for mu in range(1,epoch+1):
                self.backward_pass(X,Y,loss=loss,activation=activation,lla=lla) #Backward passing so that we have dw and db for one X,Y
                for i in range(len(self.weights)):
                    self.weights[i] = (1-(lr(t)*lam))*self.weights[i] - (lr(t)*self.dw[i])  #Updating weights using Weight , dw and L2 Regulizer
                    self.biases[i] = self.biases[i] - (lr(t)*self.db[i])    #Updating Biases normally
                self.losses_over_t.append(self.loss(self.activations[-1],Y))    #Appending the latest loss to the loss_over_t list for graphing

                if t%interval==0:
                    print(f"At iteration {t} and epoch {mu} we have loss {self.losses_over_t[-1]} Real {np.argmax(Y[:,-1])} Prediction {np.argmax(self.activations[-1][:,-1])}") #For seeing how model working

        return round(time.time()-Start , 3)
    #Saving Model  
    def save_model(self , file):    #Y , Ez just saving model to a "file"
        import pickle
        Data = {"W":self.weights , "B":self.biases}
        with open(file , "wb") as f:
            pickle.dump(Data , f)
    #Loading Model
    def load_model(self , file):    #Y , Ez just loading model from a "file"
        import pickle
        with open(file , "rb") as f:
            Data = pickle.load(f)
        self.weights = Data["W"]
        self.biases = Data["B"]

    def Test_NN(self , Test_ds=None , X=None , Y=None , limit=None , activation="sigmoid", lla="softmax", Sloss="cross entropy"):

        if (X is None) and (Y is None):
            X_all = np.column_stack([item[0] for item in Test_ds])  #Stacking X,Y column vise
            Y_all = np.column_stack([item[1] for item in Test_ds])
        elif Test_ds is None:
            X_all = X
            Y_all = Y

        if Sloss.lower() == "cross entropy":
            self.loss = self.Loss_CE
        elif Sloss.lower() in ["squared", "mse", "mean squared error"]:
            self.loss = self.Loss_sq


        self.test_loss = []
        if limit is not None:
            X_all = X_all[: , limit]
            Y_all = Y_all[: , limit]

        for i in range(X_all.shape[1]):
            x = X_all[: , [i]]
            y = Y_all[: , [i]]
            self.forward_pass(x , activation=activation , lla=lla, Sloss=Sloss)
            self.test_loss.append(self.loss(self.activations[-1] , y))
            if i%1000 == 0:
                print(f"At the {i} element the loss is {self.test_loss[-1]}")
        return self.test_loss

    def graph(self , show=True , test_loss = False , train_loss = False):
        if train_loss is True:
            Y = self.losses_over_t
            plt.plot(Y , color="blue", label="Train Loss")

        if test_loss is True:
            y = self.test_loss
            plt.plot(y , color="red", label="Test Loss")

        plt.ylabel("Loss")
        plt.xlabel("Each iteration over each Batch")
        plt.grid(True)
        plt.legend()
            
        if show is True:
            plt.show()
        
    #I can make 18/19, from scratch , to be exact 18.5/19, which means I can make 94%~97% of this Class from scratch, Pretty Cool Right? 

class math():

    def fibbonachi(n):
        fib = [0,1]
        i=1
        while len(fib) < n+2:
            fib.append(fib[-1]+fib[-2])
        return fib[-1]

    def HCF(a,b):
        assert a==int(a) and b==int(b)
        a , b = max(a,b) , min(a,b)

        EDL =  {"a":a , 
                "q":a//b , 
                "b":b , 
                "r":a%b}
        while EDL["r"] != 0:

            a = EDL["b"]
            b = EDL["r"]
            EDL =  {"a":a , "q":a//b , "b":b , "r":a%b}
        return EDL["b"]

    def LCM(a,b):
        return a*b/math.HCF(a,b)
            
class Convulational_NN(object):
    
    @staticmethod
    def initialize_filter(filter_rows, filter_cols, input_channels=1):

        raw_random = np.random.randn(input_channels , filter_rows, filter_cols)
        num_inputs = input_channels*filter_rows*filter_cols
        scale = np.sqrt(2/num_inputs)

        smart_filter = raw_random*scale
        return smart_filter
        
    def __init__(self, filter_layer , input_image , layers , classes , lla):
        
        self.lla = lla
        self.layers = filter_layer

        self.filters = []
        for lyr in self.layers:
            if type(lyr) == int:
                self.filters.append(lyr)
            else:
                filter_bank = []
                f_rows, f_cols, in_channels, out_filters = lyr # filter_rows , filter_columns , filter_depth , number_of_such_filters

                for _ in range(out_filters):

                    f = Convulational_NN.initialize_filter(f_rows, f_cols, in_channels)
                    filter_bank.append(f)
                self.filters.append(np.array(filter_bank))
        
        img_shape = input_image.shape if hasattr(input_image, 'shape') else input_image #Added by AI
        dummy_pass = self.forward_cpass(np.zeros(img_shape))
        self.fcnn = FF_Neural_Network([dummy_pass.size]+layers+[classes] , activation="relu")

    def pad(self, img , amount=1 ,num=0):
        padded_gray = np.pad(img, pad_width=amount, mode='constant', constant_values=num)
        return padded_gray

    def convolve(self, image , filter, stride=1, autopad=False, is_batch=False): #Fix 6

        image , filter = np.array(image) , np.array(filter)

        if autopad is True and stride == 1 and filter.shape[-2]%2 == 1:
            image = self.pad(image , (filter.shape[-2]-1)//2)

        R , C = image.shape[-2] , image.shape[-1]
        fr , fc = filter.shape[-2] , filter.shape[-1]
        convolved_img = []
        opr , opc = ((R-fr)//stride)+1 , ((C-fc)//stride)+1

        for r in range(0 , R-fr+1 , stride):

            for c in range(0 , C-fc+1 , stride):

                if image.ndim == 2:
                    img_f = image[r:r+fr , c:c+fc,]
                elif image.ndim == 3 and not is_batch:
                    img_f = image[: , r:r+fr , c:c+fc]
                else:
                    img_f = image[... , r:r+fr , c:c+fc]
                filtered_segment = np.sum(img_f * filter , axis=tuple(range(1, img_f.ndim)) if is_batch else None) #Fix 6
                convolved_img.append(filtered_segment)

        if is_batch:
            return np.moveaxis(np.array(convolved_img).reshape(opr , opc , -1) , -1 , 0) #Fix 6
        return np.array(convolved_img).reshape(opr , opc)

    def max_pooling(self, image, selected, stride=None, is_batch=False): #Fix 6
        if stride==None:
            stride = selected
    
        R , C = image.shape[-2] , image.shape[-1]
        Or , Oc = ((R-selected)//stride)+1 , ((C-selected)//stride)+1

        if image.ndim == 3 and not is_batch:
            channels = image.shape[0]
            pooled_channels = []
            for ch in range(channels):
                pooled = []
                for r in range(0, R - selected + 1, stride):
                    for c in range(0, C - selected + 1, stride):
                        pool = image[ch, r:r+selected, c:c+selected]
                        pooled.append(np.max(pool))
                pooled_channels.append(np.array(pooled).reshape(Or, Oc))
            return np.array(pooled_channels)
        elif image.ndim == 4 or (image.ndim == 3 and is_batch): #Fix 6
            batch_size , channels = image.shape[0] , image.shape[1]
            batch_pooled = []

            for b in range(batch_size):
                pooled_channels = []
                for ch in range(channels):
                    pooled = []
                    for r in range(0 , R-selected+1 , stride):
                        for c in range(0 , C-selected+1 , stride):
                            pool = image[b , ch , r:r+selected , c:c+selected]
                            pooled.append(np.max(pool))
                    pooled_channels.append(np.array(pooled).reshape(Or,Oc))
                batch_pooled.append(pooled_channels)
            return np.array(batch_pooled)
        else:
            pooled = []
            for r in range(0, R - selected + 1, stride):
                for c in range(0, C - selected + 1, stride):
                    pool = image[r:r+selected, c:c+selected]
                    pooled.append(np.max(pool))
            return np.array(pooled).reshape(Or, Oc)
    
    def relu(self, Z):
        return np.maximum(0,Z)
    def dReLU(self , Z):
        return (Z>0).astype(float)
    def loss(self, g,y):
        return (g-y)**2

    def forward_cpass(self, img):

        img = np.array(img)
        if img.ndim == 2:
            img = img[np.newaxis, ...] #Fix 6: give every single image an explicit channel axis -> (1,H,W)
        is_batch = img.ndim == 4 #Fix 6: (batch, channels, H, W)

        self.images_passing = [img]
        self.Zs = []
        self.is_batch = is_batch #Fix 6

        for lyr in self.filters:
            if type(lyr) == int:
                pooled = self.max_pooling(self.images_passing[-1] , lyr , is_batch=is_batch) #Fix 6
                self.images_passing.append(self.relu(pooled))
                self.Zs.append(pooled)
            else:
                lyr_out = []
                for f in lyr:
                    lyr_out.append(self.convolve(self.images_passing[-1] , f , is_batch=is_batch)) #Fix 6

                if not is_batch:
                    self.images_passing.append(self.relu(np.array(lyr_out)))
                    self.Zs.append(np.array(lyr_out))
                else:
                    stacked = np.array(lyr_out) # (out_filters, batch, opr, opc)
                    self.images_passing.append(self.relu(np.transpose(stacked, (1, 0, 2, 3))))
                    self.Zs.append(np.transpose(stacked , (1,0,2,3)))
        self.As = self.images_passing
        return self.images_passing[-1]
    
    def forward_pass(self, img):
        
        tensored_image = self.forward_cpass(img)
        if not self.is_batch:
            Flattened = tensored_image.reshape(-1, 1)
        else:
            # Fix 9: Ensure flattened output is (features, batch_size) for FCNN input
            batch_size = tensored_image.shape[0]
            Flattened = tensored_image.reshape(batch_size, -1).T
        self.fcnn.forward_pass(Flattened , activation="relu", lla=self.lla)
        return self.fcnn.activations[-1]
    
    def backward_pass(self , img , Y):

        FP = self.forward_pass(img)
        tensored_image = self.images_passing[-1]
        
        # Fix 9: Consistent batch reshaping
        if not self.is_batch:
            Flattened = tensored_image.reshape(-1, 1)
        else:
            batch_size = tensored_image.shape[0]
            Flattened = tensored_image.reshape(batch_size, -1).T
            
        self.fcnn.backward_pass(Flattened , Y=Y , activation="relu" , fowardpass=True, lla=self.lla) #Fix 1

        self.dw_fcnn , self.db_ffnn = self.fcnn.dw , self.fcnn.db

        # Fix 10: Align dZ gradient shape to (batch_size, features)
        dA_L = self.fcnn.weights[0].T @ self.fcnn.dZ_1
        if self.is_batch:
            dA_L = dA_L.T # Convert (features, batch) back to (batch, features)
        
        dZ = self.dReLU(self.Zs[-1]) * (dA_L.reshape(self.images_passing[-1].shape))

        self.dK = []
        for i in range(len(self.filters)-1 , -1 , -1):
            current_layer = self.filters[i]
            inc = self.As[i]
            
            if type(current_layer) == int:
                d_pool = np.zeros_like(inc)
                stride = current_layer

                if not self.is_batch: #Fix 6
                    for ch in range(inc.shape[0]):
                        for r in range(0 , inc.shape[1]-stride+1 , stride):
                            for c in range(0 , inc.shape[2]-stride+1 , stride):

                                patch = inc[ch , r:r+stride , c:c+stride]
                                max_r , max_c = np.unravel_index(np.argmax(patch) , patch.shape)
                                d_pool[ch , r+max_r , c+max_c] = dZ[ch , r//stride , c//stride]
                else:
                    for b in range(inc.shape[0]):
                        for ch in range(inc.shape[1]):
                            for r in range(0 , inc.shape[2]-stride+1 , stride):
                                for c in range(0 , inc.shape[3]-stride+1 , stride):

                                    patch = inc[b , ch , r:r+stride , c:c+stride]
                                    max_r , max_c = np.unravel_index(np.argmax(patch) , patch.shape)
                                    d_pool[b , ch , r+max_r , c+max_c] = dZ[b , ch , r//stride , c//stride]

                dZ = d_pool
            else:
                d_filter_bank = []
                d_inc = np.zeros_like(inc)

                if not self.is_batch: #Fix 6
                    for f_ind , f in enumerate(current_layer):
                        df = np.zeros_like(f)
                        for ch in range(f.shape[0]):
                            df[ch] = self.convolve(inc[ch] , dZ[f_ind] , stride=1)
                        d_filter_bank.append(df)

                        padded = self.pad(dZ[f_ind] , amount=f.shape[2]-1)
                        flipped = np.flip(f , axis=(1,2))

                        for ch in range(f.shape[0]):
                            d_inc[ch] += self.convolve(padded , flipped[ch] , stride=1)
                else:
                    for f_ind , f in enumerate(current_layer):
                        df = np.zeros_like(f)
                        for ch in range(f.shape[0]):
                            df[ch] = sum(self.convolve(inc[b, ch] , dZ[b, f_ind] , stride=1) for b in range(inc.shape[0]))
                        d_filter_bank.append(df)

                        flipped = np.flip(f , axis=(1,2))
                        for b in range(inc.shape[0]):
                            padded = self.pad(dZ[b, f_ind] , amount=f.shape[2]-1)
                            for ch in range(f.shape[0]):
                                d_inc[b, ch] += self.convolve(padded , flipped[ch] , stride=1)

                self.dK.append(np.array(d_filter_bank))
                dZ = d_inc
            if i > 0:
                dZ = dZ * self.dReLU(self.Zs[i-1])
        self.dK = self.dK[::-1]
        return self.fcnn.Loss_CE(FP , Y)
    def lr_k(self, t , k=0.01):
        return k

    def Train_CNN(self, Data , Validation_Set , file , batch=1 , lr=None , iterations=1, interval=1, autoload=True , graph=True , stacked=False , lam=0):
        
        if graph is True:
            plt.ion()
            
        if stacked is False:
            X = np.array([i[0] for i in Data] , ndmin=3)
            Y = np.array([i[1] for i in Data] , ndmin=2)
        else:
            X = Data[0]
            Y = Data[1]
        self.loss_over_t = []
        self.training_loss = []

        valx , valy = [v[0] for v in Validation_Set] , [v[1] for v in Validation_Set]
        valy = np.column_stack(valy) #Fix 3

        if lr is None:
            lr = self.lr_k
        
        #self.fcnn.Train_NN(X , Y , activation="relu", lr=lr)

        for t in range(1,iterations+1):

            rand_indeces = np.random.choice(len(X), size=batch , replace=False)
            if batch == 1:
                x = X[rand_indeces[0]]
                y = Y[rand_indeces[0]].reshape(-1, 1) #Fix 2
            else:
                if stacked is False:
                    x = np.array([X[i] for i in rand_indeces])[:, np.newaxis, :, :] #Fix 8
                    y = np.column_stack([Y[i] for i in rand_indeces])
                else:
                    x = X[rand_indeces]
                    y = Y[rand_indeces]
            
            batch_loss = self.backward_pass(x, y)

            for i in range(len(self.fcnn.weights)):
                self.fcnn.weights[i] -= lr(t) * self.dw_fcnn[i]
                self.fcnn.biases[i] -= lr(t) * self.db_ffnn[i]
                
            grad_idx = 0
            for i in range(len(self.filters)):
                if type(self.filters[i]) != int: 
                    avg_dK = self.dK[grad_idx] / batch
                    
                    self.filters[i] -= lr(t) * avg_dK
                    grad_idx += 1

            

            #self.loss_over_t.append(np.mean(self.fcnn.Loss_CE(self.forward_pass(valx) , valy) , axis=1))    
            val_preds = np.column_stack([self.forward_pass(v) for v in valx])  # Made by AI
            batch_val_loss = self.fcnn.Loss_CE(val_preds, valy)/len(val_preds)            
            self.loss_over_t.append(batch_val_loss)


            if t%interval == 0 or t == iterations or t==1:
                print(f"At interval {t} the Validation Loss is {self.loss_over_t[-1]}")

                if autoload is True and t%150 == 0:
                    self.save_model(file=file)
                if graph == True:
                    plt.clf()
                    x_axis = list(range(1,len(self.loss_over_t)+1))
                    plt.plot(x_axis , self.loss_over_t)

                    plt.grid(True)
                    plt.pause(0.01)
        return self.loss_over_t
    
    def save_model(self, file):
        import pickle
        Data = {"L":self.layers, "K":self.filters, "nnw":self.fcnn.weights, "nnb":self.fcnn.biases}
        with open(file , "wb") as f:
            pickle.dump(Data , f)
    def load_model(self, file):
        import pickle
        with open(file , "rb") as f:
            Data = pickle.load(f)
        self.filters = Data["K"]
        self.layers = Data["L"]
        self.fcnn.weights = Data["nnw"]
        self.fcnn.biases = Data["nnb"]

class CNN_(object):
    
    @staticmethod
    def initialize_filter(filter_rows, filter_cols, input_channels=1):

        raw_random = np.random.randn(input_channels , filter_rows, filter_cols)
        num_inputs = input_channels*filter_rows*filter_cols
        scale = np.sqrt(2/num_inputs)

        smart_filter = raw_random*scale
        return smart_filter
        
    def __init__(self, filter_layer , input_image , layers , classes , lla):
        
        self.lla = lla
        self.layers = filter_layer

        self.filters = []
        for lyr in self.layers:
            if type(lyr) == int:
                self.filters.append(lyr)
            else:
                filter_bank = []
                f_rows, f_cols, in_channels, out_filters = lyr # filter_rows , filter_columns , filter_depth , number_of_such_filters

                for _ in range(out_filters):

                    f = Convulational_NN.initialize_filter(f_rows, f_cols, in_channels)
                    filter_bank.append(f)
                self.filters.append(np.array(filter_bank))
        
        img_shape = input_image.shape if hasattr(input_image, 'shape') else input_image #Added by AI
        dummy_pass = self.forward_cpass(np.zeros(img_shape))
        self.fcnn = FF_Neural_Network([dummy_pass.size]+layers+[classes] , activation="relu")

    def pad(self, img , amount=1 ,num=0):
        padded_gray = np.pad(img, pad_width=amount, mode='constant', constant_values=num)
        return padded_gray

    def convolve(self, image , filter, stride=1, autopad=False, is_batch=False): #Fix 6

        image , filter = np.array(image) , np.array(filter)

        if autopad is True and stride == 1 and filter.shape[-2]%2 == 1:
            image = self.pad(image , (filter.shape[-2]-1)//2)

        R , C = image.shape[-2] , image.shape[-1]
        fr , fc = filter.shape[-2] , filter.shape[-1]
        convolved_img = []
        opr , opc = ((R-fr)//stride)+1 , ((C-fc)//stride)+1

        for r in range(0 , R-fr+1 , stride):

            for c in range(0 , C-fc+1 , stride):

                if image.ndim == 2:
                    img_f = image[r:r+fr , c:c+fc,]
                elif image.ndim == 3 and not is_batch:
                    img_f = image[: , r:r+fr , c:c+fc]
                else:
                    img_f = image[... , r:r+fr , c:c+fc]
                filtered_segment = np.sum(img_f * filter , axis=tuple(range(1, img_f.ndim)) if is_batch else None) #Fix 6
                convolved_img.append(filtered_segment)

        if is_batch:
            return np.moveaxis(np.array(convolved_img).reshape(opr , opc , -1) , -1 , 0) #Fix 6
        return np.array(convolved_img).reshape(opr , opc)

    def max_pooling(self, image, selected, stride=None, is_batch=False): #Fix 6
        if stride==None:
            stride = selected
    
        R , C = image.shape[-2] , image.shape[-1]
        Or , Oc = ((R-selected)//stride)+1 , ((C-selected)//stride)+1

        if image.ndim == 3 and not is_batch:
            channels = image.shape[0]
            pooled_channels = []
            for ch in range(channels):
                pooled = []
                for r in range(0, R - selected + 1, stride):
                    for c in range(0, C - selected + 1, stride):
                        pool = image[ch, r:r+selected, c:c+selected]
                        pooled.append(np.max(pool))
                pooled_channels.append(np.array(pooled).reshape(Or, Oc))
            return np.array(pooled_channels)
        elif image.ndim == 4 or (image.ndim == 3 and is_batch): #Fix 6
            batch_size , channels = image.shape[0] , image.shape[1]
            batch_pooled = []

            for b in range(batch_size):
                pooled_channels = []
                for ch in range(channels):
                    pooled = []
                    for r in range(0 , R-selected+1 , stride):
                        for c in range(0 , C-selected+1 , stride):
                            pool = image[b , ch , r:r+selected , c:c+selected]
                            pooled.append(np.max(pool))
                    pooled_channels.append(np.array(pooled).reshape(Or,Oc))
                batch_pooled.append(pooled_channels)
            return np.array(batch_pooled)
        else:
            pooled = []
            for r in range(0, R - selected + 1, stride):
                for c in range(0, C - selected + 1, stride):
                    pool = image[r:r+selected, c:c+selected]
                    pooled.append(np.max(pool))
            return np.array(pooled).reshape(Or, Oc)
    
    def relu(self, Z):
        return np.maximum(0,Z)
    def dReLU(self , Z):
        return (Z>0).astype(float)
    def loss(self, g,y):
        return (g-y)**2

    def forward_cpass(self, img):

        img = np.array(img)
        if img.ndim == 2:
            img = img[np.newaxis, ...] #Fix 6: give every single image an explicit channel axis -> (1,H,W)
        is_batch = img.ndim == 4 #Fix 6: (batch, channels, H, W)

        self.images_passing = [img]
        self.Zs = []
        self.is_batch = is_batch #Fix 6

        for lyr in self.filters:
            if type(lyr) == int:
                pooled = self.max_pooling(self.images_passing[-1] , lyr , is_batch=is_batch) #Fix 6
                self.images_passing.append(self.relu(pooled))
                self.Zs.append(pooled)
            else:
                lyr_out = []
                for f in lyr:
                    lyr_out.append(self.convolve(self.images_passing[-1] , f , is_batch=is_batch)) #Fix 6

                if not is_batch:
                    self.images_passing.append(self.relu(np.array(lyr_out)))
                    self.Zs.append(np.array(lyr_out))
                else:
                    stacked = np.array(lyr_out) # (out_filters, batch, opr, opc)
                    self.images_passing.append(self.relu(np.transpose(stacked, (1, 0, 2, 3))))
                    self.Zs.append(np.transpose(stacked , (1,0,2,3)))
        self.As = self.images_passing
        return self.images_passing[-1]
    
    def forward_pass(self, img):
        
        tensored_image = self.forward_cpass(img)
        if not self.is_batch: #Fix 6
            Flattened = tensored_image.reshape(-1, 1)
        else:
            batch_size = tensored_image.shape[0]
            Flattened = tensored_image.reshape(batch_size, -1).T
        self.fcnn.forward_pass(Flattened , activation="relu", lla=self.lla)

        return self.fcnn.activations[-1]
    
    def backward_pass(self , img , Y):

        FP = self.forward_pass(img)

        tensored_image = self.images_passing[-1]
        if not self.is_batch: #Fix 6
            Flattened = tensored_image.reshape(-1, 1)
        else:
            batch_size = tensored_image.shape[0]
            Flattened = tensored_image.reshape(batch_size, -1).T
        self.fcnn.backward_pass(Flattened , Y=Y , activation="relu" , fowardpass=True, lla=self.lla) #Fix 1

        self.dw_fcnn , self.db_ffnn = self.fcnn.dw , self.fcnn.db

        dA_L = self.fcnn.weights[0].T@self.fcnn.dZ_1
        dZ = self.dReLU(self.Zs[-1])*(dA_L.reshape(self.images_passing[-1].shape))

        self.dK = []
        for i in range(len(self.filters)-1 , -1 , -1):
            current_layer = self.filters[i]
            inc = self.As[i]
            
            if type(current_layer) == int:
                d_pool = np.zeros_like(inc)
                stride = current_layer

                if not self.is_batch: #Fix 6
                    for ch in range(inc.shape[0]):
                        for r in range(0 , inc.shape[1]-stride+1 , stride):
                            for c in range(0 , inc.shape[2]-stride+1 , stride):

                                patch = inc[ch , r:r+stride , c:c+stride]
                                max_r , max_c = np.unravel_index(np.argmax(patch) , patch.shape)
                                d_pool[ch , r+max_r , c+max_c] = dZ[ch , r//stride , c//stride]
                else:
                    for b in range(inc.shape[0]):
                        for ch in range(inc.shape[1]):
                            for r in range(0 , inc.shape[2]-stride+1 , stride):
                                for c in range(0 , inc.shape[3]-stride+1 , stride):

                                    patch = inc[b , ch , r:r+stride , c:c+stride]
                                    max_r , max_c = np.unravel_index(np.argmax(patch) , patch.shape)
                                    d_pool[b , ch , r+max_r , c+max_c] = dZ[b , ch , r//stride , c//stride]

                dZ = d_pool
            else:
                d_filter_bank = []
                d_inc = np.zeros_like(inc)

                if not self.is_batch: #Fix 6
                    for f_ind , f in enumerate(current_layer):
                        df = np.zeros_like(f)
                        for ch in range(f.shape[0]):
                            df[ch] = self.convolve(inc[ch] , dZ[f_ind] , stride=1)
                        d_filter_bank.append(df)

                        padded = self.pad(dZ[f_ind] , amount=f.shape[2]-1)
                        flipped = np.flip(f , axis=(1,2))

                        for ch in range(f.shape[0]):
                            d_inc[ch] += self.convolve(padded , flipped[ch] , stride=1)
                else:
                    for f_ind , f in enumerate(current_layer):
                        df = np.zeros_like(f)
                        for ch in range(f.shape[0]):
                            df[ch] = sum(self.convolve(inc[b, ch] , dZ[b, f_ind] , stride=1) for b in range(inc.shape[0]))
                        d_filter_bank.append(df)

                        flipped = np.flip(f , axis=(1,2))
                        for b in range(inc.shape[0]):
                            padded = self.pad(dZ[b, f_ind] , amount=f.shape[2]-1)
                            for ch in range(f.shape[0]):
                                d_inc[b, ch] += self.convolve(padded , flipped[ch] , stride=1)

                self.dK.append(np.array(d_filter_bank))
                dZ = d_inc
            if i > 0:
                dZ = dZ * self.dReLU(self.Zs[i-1])
        self.dK = self.dK[::-1]
    def lr_k(self, t , k=0.01):
        return k

    def Train_CNN(self, Data , Validation_Set , file , batch=1 , lr=None , iterations=1, interval=1, graph=True, autoload=True):

        if graph is True:
            plt.ion()

        X = np.array([i[0] for i in Data] , ndmin=3)
        Y = np.array([i[1] for i in Data] , ndmin=2)
        self.loss_over_t = []

        valx , valy = [v[0] for v in Validation_Set] , [v[1] for v in Validation_Set]
        valy = np.column_stack(valy) #Fix 3

        if lr is None:
            lr = self.lr_k
        
        #self.fcnn.Train_NN(X , Y , activation="relu", lr=lr)

        for t in range(1,iterations+1):

            rand_indeces = np.random.choice(len(X), size=batch , replace=False)
            if batch == 1:
                x = X[rand_indeces[0]]
                y = Y[rand_indeces[0]].reshape(-1, 1) #Fix 2
            else:
                x = np.array([X[i] for i in rand_indeces])[:, np.newaxis, :, :] #Fix 8
                y = np.column_stack([Y[i] for i in rand_indeces])
            
            self.backward_pass(x, y)

            for i in range(len(self.fcnn.weights)):
                self.fcnn.weights[i] -= lr(t) * self.dw_fcnn[i]
                self.fcnn.biases[i] -= lr(t) * self.db_ffnn[i]
                
            grad_idx = 0
            for i in range(len(self.filters)):
                if type(self.filters[i]) != int: 
                    avg_dK = self.dK[grad_idx] / batch
                    
                    self.filters[i] -= lr(t) * avg_dK
                    grad_idx += 1

            #self.loss_over_t.append(np.mean(self.fcnn.Loss_CE(self.forward_pass(valx) , valy) , axis=1))    
            val_preds = np.column_stack([self.forward_pass(v) for v in valx])  # Made by AI
            batch_val_loss = self.fcnn.Loss_CE(val_preds, valy)            
            self.loss_over_t.append(batch_val_loss)


            if t%interval == 0 or t == iterations or t==1:
                print(f"At iteration {t} the Validation Loss is {self.loss_over_t[-1]}")
                if autoload is True:
                    self.load_model(file=file)
                if graph == True:
                    plt.clf()
                    x_axis = list(range(1,len(self.loss_over_t)+1))
                    plt.plot(x_axis , self.loss_over_t)
                    plt.grid(True)
                    plt.pause(0.01)

        return self.loss_over_t

class S2S_RNN_m2m(object):

    def __init__(self  , inp_r , s_len , vocab_size , opt_r=None , factor=0.01):

        if opt_r is None:
            op = inp_r
        self.ip , self.op = inp_r , op
        self.s_r = s_len

        #State transmitters
        self.Wsx = np.array(np.random.randn(self.s_r , self.ip) , ndmin=2)*factor
        self.Wss = np.array(np.random.randn(self.s_r , self.s_r) , ndmin=2)*factor
        self.bs = np.array(np.zeros((self.s_r , 1)) , ndmin=2).reshape(-1,1)

        #st = f1(Wsx@xt + Wss@st-1 + bs)

        #Output generators
        self.Wo = np.array(np.random.randn(vocab_size , self.s_r) , ndmin=2)*factor
        self.bo = np.array(np.zeros((vocab_size , 1)) , ndmin=2)

        #zt = Wo@st + bo --> yt = softmax(zt)
    
    def softmax(self , distribution):

        shifted = distribution - np.max(distribution)
        shifted = np.clip(shifted, -500, 500)
        exps = np.exp(shifted)
        min_subbed = np.sum(exps, axis=0, keepdims=True)

        return exps / (min_subbed + 1e-15)
    def same(self , z):
        return z

    def tanh(self , z):
        return np.tanh(z)
    def ReLU(self , z):
        return np.maximum(0 , z)
    def sigmoid(self , z):
        return (1/(1+np.exp(-z)))

    def dReLU(self , z):
        return np.where(z>0 , 1 , 0)
    def dtanh(self , z):
        return 1-(self.tanh(z)**2)
    def dSigmoid(self , z):
        S = (self.sigmoid(z))
        return S*(1-S)
    
    def forward_pass(self , X , Y=None , s0=None , f1_0="relu" , f2_0="softmax" , loss=None):

        func_map1 = {"relu":self.ReLU , "tanh":self.tanh , "sigmoid":self.sigmoid}
        func_map2 = {"softmax":self.softmax , "same":self.same , "linear":self.same}
        loss_dict = {"cross entropy":self.Loss_CE , "mse":self.Loss_MSE , "squared":self.Loss_MSE}

        self.f1 , self.f2 = func_map1.get(f1_0 , self.ReLU) , func_map2.get(f2_0 , self.softmax)

        #So X is a sequence containing X = {x1,x2,x3...xT}
        
        if s0 is None:
            self.s = np.array(np.zeros((self.s_r , 1)) , ndmin=2) #s0
        else:
            self.s = np.array(s0 , ndmin=2).reshape(self.s_r , 1)
        self.P = []
        self.states = [self.s.copy()]
        self.Zs = []
        self.hidden_Zs = []
        self.losses_over_t = []
        self.Loss_f = loss_dict.get(loss , None)

        for t , x_t in enumerate(list(X)):

            x_t = np.array(x_t , ndmin=2).reshape(-1,1)

            #Hidden Pre Activation
            z_hidden = self.Wsx@x_t + self.Wss@self.s + self.bs
            self.hidden_Zs.append(z_hidden)

            #Now calculation transmitting sequence

            self.s = self.f1(self.Wsx@x_t + self.Wss@self.s + self.bs) # Using the recurrence formula
            
            self.states.append(self.s.copy())

            #Calculating the Output
            self.z = self.Wo@self.s + self.bo #This is z_t

            self.P.append(self.f2(self.z)) #This is y_t being appended to output seq Y
            self.Zs.append(self.z)  

            if self.Loss_f is not None and Y is not None:
                self.losses_over_t.append(np.sum(self.Loss_f(self.P[t],Y[t])))     

        return self.P

    def Loss_CE(self , p , y):
        p = np.clip(p , 1e-15 , 1)
        return -y*np.log(p)
    def Loss_MSE(self , p , y):
        return (y-p)**2

    def dLoss_CE(self , p , y):
        return -y/p
    def dLoss_MSE(self , p , y):
        return 2*(y-p)

    def One_Hot(self , ind , l):
        arr = np.zeros((l,1))
        arr[0 , ind] = 1
        return arr
    
    def backward_pass(self , X , Y , for_pass=True , f1="relu" , f2="softmax" , loss="cross entropy" , s0=None , one_hot=False):

        deriv_map = {self.ReLU : self.dReLU , self.sigmoid:self.dSigmoid , self.tanh:self.dtanh}

        if for_pass:
            self.forward_pass(X=X , Y=Y , s0=s0 , f1_0=f1 , f2_0=f2 , loss=loss)
            self._loss_ = np.mean(self.losses_over_t)

        df1 = deriv_map[self.f1]

        if loss.lower() in ["cross entropy", "nll", "mse"]: #Initialization of Weights
            eval_steps = min(len(self.P) , len(Y))
            if not one_hot:
                delta_0 = [self.P[t] - np.array(Y[t] , ndmin=2).reshape(-1,1) for t in range(eval_steps)] # idk why this comment is green
            else:
                delta_0 = [self.P[t] - self.One_Hot( ind=Y[t] , l=len(self.P[t]) ) for t in range(eval_steps)]
        else:
            raise ValueError(f"Unsupported function type : {loss}")

        #Initialization ARC

        dWo , dbo = np.zeros_like(self.Wo) , np.zeros_like(self.bo) #Initialize the w,b for the output func

        dWss , dWsx , dbs = np.zeros_like(self.Wss) , np.zeros_like(self.Wsx) , np.zeros_like(self.bs) #Innitialize w, for the Recurrence Function

        ds_1 = np.zeros( (self.s_r , 1) )

        T = len(X)

        for t in range(T-1 , -1 , -1):
            
            x_t = np.array(X[t] , ndmin=2).reshape(-1,1)

            delta_o = delta_0[t].copy()

            dWo += delta_o@self.states[t+1].T
            dbo += delta_o.copy()

            ds = self.Wo.T@delta_o + ds_1

            s = self.states[t+1]
            z = self.hidden_Zs[t]
            delta_s = ds*df1(z)

            dWsx += delta_s @ x_t.T
            dWss += delta_s @ self.states[t].T
            dbs += delta_s.copy()

            ds_1 = self.Wss.T @ delta_s

        self.dWo , self.dbo = dWo.copy() , dbo.copy()
        self.dWss , self.dWsx , self.dbs = dWss.copy() , dWsx.copy() , dbs.copy()

        return (self.dWo , self.dbo , self.dWss , self.dWsx , self.dbs) , self._loss_

    def Train_RNN(self , X , Y , epoch=5 , iterations=10000 , lr=lambda t: 0.01 , f1="relu" , f2="same" , loss="mse" , s0=None , one_hot=False , for_pass=True , stocastic=False , interval=100):

        if stocastic:
            for ep in range(epoch):
                for t in range(1,iterations+1):

                    k = np.random.choice(len(X))
                    T = ep*iterations + t

                    derivs , _loss_ = self.backward_pass(X[k] , Y[k] , for_pass=for_pass , f1=f1 , f2=f2 , loss=loss , s0=s0 , one_hot=one_hot)

                    self.Wo , self.bo = self.Wo - (lr(T)*(derivs[0])) , self.bo - (lr(T)*(derivs[1]))
                    self.Wss , self.Wsx , self.bs = self.Wss - (lr(T)*(derivs[2])) , self.Wsx - (lr(T)*(derivs[3])) , self.bs - (lr(T)*(derivs[4]))
                    if t%50 == 0:
                        print(f"Loss at {t} : {_loss_}")
            return True
        elif not stocastic:

            for ep in range(1,1+epoch):
                for k in range(len(X)):
                    t = k

                    derivs , _loss_ = self.backward_pass(X[k] , Y[k] , for_pass=for_pass , f1=f1 , f2=f2 , loss=loss , s0=s0 , one_hot=one_hot)

                    self.Wo , self.bo = self.Wo - (lr(t)*(derivs[0])) , self.bo - (lr(t)*(derivs[1]))
                    self.Wss , self.Wsx , self.bs = self.Wss - (lr(t)*(derivs[2])) , self.Wsx - (lr(t)*(derivs[3])) , self.bs - (lr(t)*(derivs[4]))
                    if t%interval == 0:
                        print(f"Loss at {t} : {_loss_}")
            return True

    def save_model(self , file):
        Data = (self.Wo , self.bo , self.Wsx , self.Wss , self.bs)

        with open(file , "wb") as f:
            pickle.dump(Data , f)
        return 1

    def load_model(self , file):

        with open(file , "rb") as f:
            Data = pickle.load(f)

        self.Wo , self.bo , self.Wsx , self.Wss , self.bs = Data

        return 1

if __name__ == "__main__":

    pass
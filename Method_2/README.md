In this method, we first visualize the data and apply some preprocessing steps.
[eda.py](https://github.com/ZER-0-NE/OrcaCNN-Demo/blob/master/Method_2/eda.py) shows the code for plotting.

We include a [csv](https://github.com/ZER-0-NE/OrcaCNN-Demo/blob/master/Method_2/whales.csv) file which has fname(filename) and label(Positive or Negative) as the columns and also another [csv](https://github.com/ZER-0-NE/OrcaCNN-Demo/blob/master/Method_2/whales_classlabel.csv) file with all the classes divided.

### Time Signal

![signal](assets/1.png)


### Fourier Transforms

![fft](assets/2.png)

### Filter Bank Coefficients

![fbank](assets/3.png)


### Mel Frequency Cepstrum Coefficients

![mel](assets/4.png)

All the pre-processed files are stored in [clean](https://github.com/ZER-0-NE/OrcaCNN-Demo/tree/master/Method_2/clean) directory. 

Intitially, there was a lot of class imbalance (~67% in negative class). To tackle this, we had two options, one to augment the minority classes, or to reduce the frequency of data in negative class. We reduced the frequency of negative class but I have also included [data_aug.py](https://github.com/ZER-0-NE/OrcaCNN-Demo/blob/master/Method_2/data_aug.py) if anyone is interested to augment the minority classes.

In the [model.py](https://github.com/ZER-0-NE/OrcaCNN-Demo/blob/master/Method_2/model.py) file we use two architectures, CNN and RNN specifically. They have shown promising results in the past research work with acoustic data. We train them on the 9 class labels defined in [whales_classlabel.csv](https://github.com/ZER-0-NE/OrcaCNN-Demo/blob/master/Method_2/whales_classlabel.csv). 

#### CNN Architecture


        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        conv2d_1 (Conv2D)            (None, 9, 13, 16)         160       
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 9, 13, 32)         4640      
        _________________________________________________________________
        conv2d_3 (Conv2D)            (None, 9, 13, 64)         18496     
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 4, 6, 64)          0         
        _________________________________________________________________
        dropout_1 (Dropout)          (None, 4, 6, 64)          0         
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 1536)              0         
        _________________________________________________________________
        dense_1 (Dense)              (None, 64)                98368     
        _________________________________________________________________
        dense_2 (Dense)              (None, 32)                2080      
        _________________________________________________________________
        dense_3 (Dense)              (None, 9)                 297       
        =================================================================
        Total params: 124,041
        Trainable params: 124,041
        Non-trainable params: 0



#### RNN Architecture


        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        lstm_1 (LSTM)                (None, 9, 64)             19968     
        _________________________________________________________________
        lstm_2 (LSTM)                (None, 9, 64)             33024     
        _________________________________________________________________
        dropout_1 (Dropout)          (None, 9, 64)             0         
        _________________________________________________________________
        time_distributed_1 (TimeDist (None, 9, 32)             2080      
        _________________________________________________________________
        time_distributed_2 (TimeDist (None, 9, 16)             528       
        _________________________________________________________________
        time_distributed_3 (TimeDist (None, 9, 8)              136       
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 72)                0         
        _________________________________________________________________
        dense_4 (Dense)              (None, 9)                 657       
        =================================================================
        Total params: 56,393
        Trainable params: 56,393
        Non-trainable params: 0
 

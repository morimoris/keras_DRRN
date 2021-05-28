from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Add, ReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization

def DRRN(recursive_brocks, recursive_units, input_channels, filter_num = 128, filter_size = (3, 3)): 
    """
    recursive_brocks : numbers of recursive blocks.(>=1)
    recursive_units : Number of residual units in the recursive block.(>=1)       
    input_channels : Channels of input_img.(gray → 1, RGB → 3)
    filter_num : Filter numbers.(default 128)
    filter_size : Filter size.(default 3*3)
    """
    Units_conv_1 = Conv2D(filters = filter_num, kernel_size = filter_size, padding = "same")
    Units_conv_2 = Conv2D(filters = filter_num, kernel_size = filter_size, padding = "same")
    """
<<<<<<< HEAD
    units_conv1, units_conv2 → same weights
=======
    units_conv → same weight
>>>>>>> 350c2147f1a4a0615c706e26d5306c910ea4adf0
    """
    
    #model
    input_shape = Input((None, None, input_channels))                      
    #recursive blocks
    for B in range(recursive_brocks):
        if B == 0:
            conv2d_0 = Conv2D(filters = filter_num, kernel_size = filter_size, padding = "same")(input_shape)
        else:
            conv2d_0 = Conv2D(filters = filter_num, kernel_size = filter_size, padding = "same")(add_conv)
        #a box of variables for looping.
        add_conv = conv2d_0
        #recursive units
        for U in range(recursive_units):
            unit_batch_1 = BatchNormalization()(add_conv)
            unit_relu_1 = ReLU()(unit_batch_1)
            unit_conv_1 = Units_conv_1(unit_relu_1) 
            unit_batch_2 = BatchNormalization()(unit_conv_1)
            unit_relu_2 = ReLU()(unit_batch_2)
            unit_conv_2 = Units_conv_2(unit_relu_2) 
            add_conv = Add()([conv2d_0, unit_conv_2])

    #output conv2d
    conv2d_out = Conv2D(filters = input_channels, kernel_size = filter_size, padding = "same")(add_conv)

    #skip connection
    skip_connection = Add()([input_shape, conv2d_out])
    
    model = Model(inputs = input_shape, outputs = skip_connection)
    model.summary()

    return model


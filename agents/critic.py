from keras import layers, models, optimizers, regularizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, lr=0.001, lr_decay=1e-9):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.lr_decay = lr_decay
        # Initialize any other variables here
        #print("Initializing Critic model")
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
            
        # Scale [0, 1] output for each action dimension to proper range
        states_scaled = layers.Lambda(lambda x: (x / 40), name='states_scaled')(states)
                                      
        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=64, 
                                  kernel_initializer='uniform', 
                                   bias_initializer='random_uniform',
                                  activation='relu', 
                                  kernel_regularizer=regularizers.l2(0.01))(states_scaled)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dense(units=128, 
                                  kernel_initializer='uniform', 
                                   bias_initializer='random_uniform',
                                  activation='relu', 
                                  kernel_regularizer=regularizers.l2(0.01))(net_states)
        #net_states = layers.BatchNormalization()(net_states)
        #net_states = layers.Dense(units=64, 
        #                          kernel_initializer='uniform', 
        #                           bias_initializer='random_uniform',
        #                          activation='relu', 
        #                          kernel_regularizer=regularizers.l2(0.01))(net_states)
        
        # Scale [0, 1] output for each action dimension to proper range
        actions_scaled = layers.Lambda(lambda x: ((x - 0) / 900),
                                name='actions_scaled')(actions)
        
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=64, 
                                   kernel_initializer='uniform', 
                                   bias_initializer='random_uniform', 
                                   activation='relu', 
                                   kernel_regularizer=regularizers.l2(0.01))(actions_scaled)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dense(units=128,  
                                   kernel_initializer='uniform', 
                                   bias_initializer='random_uniform',
                                   activation='relu', 
                                   kernel_regularizer=regularizers.l2(0.01))(net_actions)
        #net_actions = layers.BatchNormalization()(net_actions)
        #net_actions = layers.Dense(units=64,  
        #                           kernel_initializer='uniform',
        #                           bias_initializer='random_uniform', 
        #                           activation='relu', 
        #                           kernel_regularizer=regularizers.l2(0.01))(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr, 
                                    beta_1=0.9, 
                                    beta_2=0.999, 
                                    epsilon=None, 
                                    decay=self.lr_decay, 
                                    amsgrad=False)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

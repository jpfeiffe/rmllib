

class LocalModel:
    '''
    Basic local model implementation.  Can do iid learning to collective inference.
    '''
    def __init__(self, name='localmodel', learn_method='r_iid', infer_method='r_iid',\
                       calibrate=True, unlabeled_confidence=1, twohop_confidence=1):
        '''
        Set parameters for learning and inference
        
        :name: name to keep track of the model
        :param learn_method: 'iid' for iid learning, 'r_iid' for relational iid, 
            'r_joint' for incorporating uncertain labels, 'r_twohop' for doing two
            steps away inference
        :param infer_method: 'iid' for iid inference, 'r_iid' for relational iid inference,
            'r_join' for variational inference
        :param calibrate: after each inference step adjust the median prediction to match labeled set
        :param unlabeled_confidence: When doing inference and using uncertain neighbors how much do we trust them
        '''
        self.name = name
        self.set_learn_method(learn_method)
        self.set_infer_method(infer_method)
        self.calibrate = calibrate
        self.unlabeled_confidence = unlabeled_confidence
        self.twohop_confidence = twohop_confidence

    def set_infer_method(self, infer_method):
        self.infer_method = infer_method

    def set_learn_method(self, learn_method):
        self.learn_method = learn_method

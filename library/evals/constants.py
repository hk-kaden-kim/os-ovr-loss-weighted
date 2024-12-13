
class smallscale():
    def __init__(self):

        self.sm_data_info = [
            {'info':['smallscale', 'LeNet_neg_0', 'SoftMax']},
        ]

        self.eos_data_info = [
            {'info':['smallscale', 'LeNet_neg_10k', 'EOS']}, {'info':['smallscale', 'LeNet_neg_20k', 'EOS']},
            {'info':['smallscale', 'LeNet_neg_30k', 'EOS']}, {'info':['smallscale', 'LeNet_neg_All', 'EOS']},
        ]

        self.ovr_data_info = {
            'base': [{'info':['smallscale', 'LeNet_neg_0', 'OvR']}],
            'C': [{'info':['smallscale', 'LeNet_C_neg_0_b', 'OvR']},{'info':['smallscale', 'LeNet_C_neg_0_g', 'OvR']}],
            'F': [{'info':['smallscale', 'LeNet_F_neg_0_02', 'OvR']},{'info':['smallscale', 'LeNet_F_neg_0_06', 'OvR']},
                {'info':['smallscale', 'LeNet_F_neg_0_1', 'OvR']},{'info':['smallscale', 'LeNet_F_neg_0_2', 'OvR']},
                {'info':['smallscale', 'LeNet_F_neg_0_3', 'OvR']}],
            'H': [{'info':['smallscale', 'LeNet_H_neg_0_0', 'OvR']},{'info':['smallscale', 'LeNet_H_neg_0_02', 'OvR']},
                {'info':['smallscale', 'LeNet_H_neg_0_04', 'OvR']},{'info':['smallscale', 'LeNet_H_neg_0_06', 'OvR']},
                {'info':['smallscale', 'LeNet_H_neg_0_08', 'OvR']}],
        }

        self.ovr_data_info_neg = {
            'base': [{'info':['smallscale', 'LeNet_neg_All', 'OvR']}],
            'C': [{'info':['smallscale', 'LeNet_C_neg_All_b', 'OvR']},{'info':['smallscale', 'LeNet_C_neg_All_g', 'OvR']}],
            'F': [{'info':['smallscale', 'LeNet_F_neg_All_02', 'OvR']},{'info':['smallscale', 'LeNet_F_neg_All_06', 'OvR']},
                {'info':['smallscale', 'LeNet_F_neg_All_1', 'OvR']},{'info':['smallscale', 'LeNet_F_neg_All_2', 'OvR']},
                {'info':['smallscale', 'LeNet_F_neg_All_3', 'OvR']}],
            'H': [{'info':['smallscale', 'LeNet_H_neg_All_0', 'OvR']},{'info':['smallscale', 'LeNet_H_neg_All_02', 'OvR']},
                {'info':['smallscale', 'LeNet_H_neg_All_04', 'OvR']},{'info':['smallscale', 'LeNet_H_neg_All_06', 'OvR']},
                {'info':['smallscale', 'LeNet_H_neg_All_08', 'OvR']}],
        }

        self.ovr_data_info_10k = {
            'base':[{'info':['smallscale', 'LeNet_neg_10k', 'OvR']}],
            'C':[{'info':['smallscale', 'LeNet_C_neg_10k_b', 'OvR']},],
            'F':[{'info':['smallscale', 'LeNet_F_neg_10k_1', 'OvR']},],
            'H':[{'info':['smallscale', 'LeNet_H_neg_10k_02', 'OvR']},],
        }
        self.ovr_data_info_20k = {
            'base':[{'info':['smallscale', 'LeNet_neg_20k', 'OvR']}],
            'C':[{'info':['smallscale', 'LeNet_C_neg_20k_b', 'OvR']},],
            'F':[{'info':['smallscale', 'LeNet_F_neg_20k_1', 'OvR']},],
            'H':[{'info':['smallscale', 'LeNet_H_neg_20k_02', 'OvR']},],
        }
        self.ovr_data_info_30k = {
            'base':[{'info':['smallscale', 'LeNet_neg_30k', 'OvR']}],
            'C':[{'info':['smallscale', 'LeNet_C_neg_30k_b', 'OvR']},],
            'F':[{'info':['smallscale', 'LeNet_F_neg_30k_1', 'OvR']},],
            'H':[{'info':['smallscale', 'LeNet_H_neg_30k_02', 'OvR']},],
        }

class largescale():
    def __init__(self):
        self.sm_data_info = [
                {'info':["largescale_1", 'ResNet_50_neg_0', 'SoftMax']},
                {'info':["largescale_2", 'ResNet_50_neg_0', 'SoftMax']},
                {'info':["largescale_3", 'ResNet_50_neg_0', 'SoftMax']},
        ]

        self.eos_data_info = [
                {'info':["largescale_1", 'ResNet_50_neg_All', 'EOS']},
                {'info':["largescale_2", 'ResNet_50_neg_All', 'EOS']},
                {'info':["largescale_3", 'ResNet_50_neg_All', 'EOS']},
        ]

        self.ovr_data_info_tune = {
        'C':[{'info':["largescale_2", 'ResNet_50_C_neg_0_b', 'OvR']},{'info':["largescale_2", 'ResNet_50_C_neg_0_g', 'OvR']}],
        'F':[{'info':["largescale_2", 'ResNet_50_F_neg_0_1', 'OvR']},{'info':["largescale_2", 'ResNet_50_F_neg_0_2', 'OvR']},{'info':["largescale_2", 'ResNet_50_F_neg_0_3', 'OvR']}],
        'H':[{'info':["largescale_2", 'ResNet_50_H_neg_0_02', 'OvR']},{'info':["largescale_2", 'ResNet_50_H_neg_0_06', 'OvR']}],
        }

        self.ovr_data_info_neg_tune = {
        'C':[{'info':["largescale_2", 'ResNet_50_C_neg_All_b', 'OvR']},{'info':["largescale_2", 'ResNet_50_C_neg_All_g', 'OvR']}],
        'F':[{'info':["largescale_2", 'ResNet_50_F_neg_All_1', 'OvR']},{'info':["largescale_2", 'ResNet_50_F_neg_All_2', 'OvR']},{'info':["largescale_2", 'ResNet_50_F_neg_All_3', 'OvR']}],
        'H':[{'info':["largescale_2", 'ResNet_50_H_neg_All_02', 'OvR']},{'info':["largescale_2", 'ResNet_50_H_neg_All_06', 'OvR']}],
        }

        self.ovr_data_info = {
        'base':[{'info':["largescale_1", 'ResNet_50_neg_0', 'OvR']},
                {'info':["largescale_2", 'ResNet_50_neg_0', 'OvR']},
                {'info':["largescale_3", 'ResNet_50_neg_0', 'OvR']},],
        'C':[{'info':["largescale_1", 'ResNet_50_C_neg_0_g', 'OvR']},
            {'info':["largescale_2", 'ResNet_50_C_neg_0_g', 'OvR']},
            {'info':["largescale_3", 'ResNet_50_C_neg_0_g', 'OvR']},],
        'F':[{'info':["largescale_1", 'ResNet_50_F_neg_0_1', 'OvR']},
            {'info':["largescale_2", 'ResNet_50_F_neg_0_1', 'OvR']},
            {'info':["largescale_3", 'ResNet_50_F_neg_0_1', 'OvR']},],
        'H':[{'info':["largescale_1", 'ResNet_50_H_neg_0_02', 'OvR']},
            {'info':["largescale_2", 'ResNet_50_H_neg_0_02', 'OvR']},
            {'info':["largescale_3", 'ResNet_50_H_neg_0_02', 'OvR']},],
        }

        self.ovr_data_info_neg = {
        'base':[{'info':["largescale_1", 'ResNet_50_neg_All', 'OvR']},
                {'info':["largescale_2", 'ResNet_50_neg_All', 'OvR']},
                {'info':["largescale_3", 'ResNet_50_neg_All', 'OvR']},],
        'C':[{'info':["largescale_1", 'ResNet_50_C_neg_All_b', 'OvR']},
            {'info':["largescale_2", 'ResNet_50_C_neg_All_b', 'OvR']},
            {'info':["largescale_3", 'ResNet_50_C_neg_All_b', 'OvR']},],
        'F':[{'info':["largescale_1", 'ResNet_50_F_neg_All_3', 'OvR']},
            {'info':["largescale_2", 'ResNet_50_F_neg_All_3', 'OvR']},
            {'info':["largescale_3", 'ResNet_50_F_neg_All_3', 'OvR']},],
        'H':[{'info':["largescale_1", 'ResNet_50_H_neg_All_02', 'OvR']},
            {'info':["largescale_2", 'ResNet_50_H_neg_All_02', 'OvR']},
            {'info':["largescale_3", 'ResNet_50_H_neg_All_02', 'OvR']},],
        }